"""ASH CLI: doctor, run, eval.

The commands mirror the ASH cycle:

    ash doctor        environment self-check (CDP, game, GPU, corpus)
    ash run           inference rounds with automatic bootstrapping; K is fit
                      on the corpus at startup, then refit on D^R each round
    ash eval          print a loop report as JSON

Every command works with --backend fake for offline dry runs; the CDP backend
is only required for real play.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("ash")


def _setup_logging(verbose: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    from ash.env.base import make_env

    checks: list[tuple[str, str]] = []
    try:
        import torch

        checks.append(("torch", torch.__version__))
        checks.append(("cuda-available", str(torch.cuda.is_available())))
    except ImportError:
        checks.append(("torch", "MISSING (pip install .[train])"))
    try:
        import hdbscan  # noqa: F401

        checks.append(("hdbscan", "ok"))
    except ImportError:
        checks.append(("hdbscan", "MISSING"))
    try:
        import cv2

        checks.append(("opencv", cv2.__version__))
    except ImportError:
        checks.append(("opencv", "MISSING"))
    try:
        env = _make_env(args.backend)
        obs = env.reset()
        checks.append((f"backend:{args.backend}", f"obs {obs.rgb.shape}"))
        env.close()
    except Exception as e:  # pragma: no cover - diagnostic path
        checks.append((f"backend:{args.backend}", f"FAIL: {e}"))
    for name, status in checks:
        print(f"  {name:<24} {status}")
    return 0


def _make_env(backend: str):
    """Construct a backend, capturing at the embedder's canonical frame size.

    The CDP backend defaults to a 128x128 capture, but the corpus is extracted
    at DEFAULT_IMAGE_SIZE and DINOv2 features depend on the resolution they are
    computed at - so a smaller capture would make every live frame
    systematically unlike the corpus K was fit on.  Backends that own their
    observation size (the fake one) take no resize argument.
    """
    from ash.env.base import make_env
    from ash.memory.embeddings import DEFAULT_IMAGE_SIZE

    if backend == "cdp":
        # drive="realtime": never take the engine's loop away from it.  Measured
        # with the frame pump, this game's hurt state pins at _pRealState 6 and a
        # trap that teleports correctly under the engine's own loop never
        # resolves - so a pumped self-play run collects trajectories of a
        # character that cannot move.  Determinism is not worth that here; it is
        # needed for replay/search, not for gathering behaviour.
        return make_env(
            backend,
            resize=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            drive="realtime",
        )
    return make_env(backend)


def _check_corpus_interval(corpus_dir: str | None, expected_fps: float) -> str | None:
    """Complain when the corpus was sampled at a different rate than we act at.

    This is the failure that wasted the first live rounds: the agent stepped one
    game frame (1/60 s) while the corpus held frames 2 s apart, so the IDM was
    trained on one time scale and applied at another and collapsed to a constant
    class.  Nothing in the logs looked wrong.  Check the manifest and say so.
    """
    manifest = Path(corpus_dir or "data/corpus") / "manifest.json"
    if not manifest.exists():
        return None
    try:
        entries = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    seen = {float(e["fps"]) for e in entries if isinstance(e, dict) and "fps" in e}
    if not seen:
        return None
    if any(abs(fps - expected_fps) > 1e-6 for fps in seen):
        return (
            "corpus under %s was sampled at %s fps but the control interval "
            "(%.3f s) implies %.3f fps; the IDM is trained on the agent's "
            "timestep and applied to the corpus, so these must match "
            "(re-extract with scripts/build_corpus.py --fps %g)"
            % (corpus_dir or "data/corpus", sorted(seen),
               expected_fps and 1.0 / expected_fps, expected_fps, expected_fps)
        )
    return None


def cmd_run(args: argparse.Namespace) -> int:
    from ash.config import load_game_config
    from ash.loop.orchestrator import LoopConfig, Orchestrator, load_or_init_idm, load_or_init_policy
    from ash.loop.runner import InferenceRunner
    from ash.utils.device import resolve_device

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    game = load_game_config()
    log.info(
        "control interval %.3f s -> %d game frames/action, corpus %.3f fps",
        game.control_interval_s, game.control_frame_skip, game.corpus_fps,
    )
    if args.backend == "cdp":
        # Say which drive mode the round will use: the difference is not
        # cosmetic (a hand-driven ticker pins this game's hurt state and the
        # trap never resolves), so it must be visible in every run's log.
        log.info("live drive mode: realtime (engine keeps its own ticker)")
    mismatch = _check_corpus_interval(args.corpus, game.corpus_fps)
    if mismatch:
        log.error("corpus/agent time scales disagree: %s", mismatch)
        print("语料与 agent 时间尺度不一致，拒绝启动：\n  %s" % mismatch, file=sys.stderr)
        return 4

    # The action space is the single source of truth for every action-shaped
    # tensor: the policy head width, the IDM head width and the index->mask
    # mapping must all agree on it.  Deriving them here (rather than letting
    # each model default independently) is what keeps three different counts
    # from silently coexisting again.
    probe_env = _make_env(args.backend)
    try:
        action_space = probe_env.action_space
        # Fail fast on a live run rather than letting the loop discover it.  In
        # RPG Maker the ok/cancel keys ARE the keys the policy uses for
        # jump/attack, so the very first thing a live run must not do is press
        # one on a title or menu screen - that is how an unattended run loaded
        # the player's save.
        if args.backend == "cdp":
            reason = probe_env.unsafe_reason()
            if reason and _is_menu_scene(probe_env):
                # An agent that walked into a menu can back out with cancel on
                # its own, so a previous run's accident is not a reason to refuse
                # to start.  This presses cancel ONLY (never ok, never a
                # direction) and only on a scene agent.js lists as a menu.
                escaped = probe_env.escape_menu(max_presses=6)
                log.info("menu escape before start: %s", escaped)
                reason = probe_env.unsafe_reason()
            if reason and not probe_env.is_confirm_scene():
                log.error("refusing to start a live run: %s", reason)
                print(
                    "拒绝启动实机循环：%s\n"
                    "请先在游戏里读档进入可操作的地图，再重试。" % reason,
                    file=sys.stderr,
                )
                return 3
            if reason:
                # An allow-listed confirm screen clears with a single ok, so the
                # round can start here instead of demanding manual intervention.
                log.info("starting from a confirm scene (one ok will clear it): %s", reason)
    finally:
        probe_env.close()
    log.info("action space: %d masks", len(action_space))

    policy = load_or_init_policy(args.policy, _policy_config(args, len(action_space)))
    idm = load_or_init_idm(args.idm, _idm_config(args, len(action_space)))

    from ash.memory.embeddings import FrameEmbedder

    device = str(resolve_device(args.device))
    embedder = FrameEmbedder(device)

    # The corpus is embedded exactly once and the result serves both consumers:
    # K is fit on these embeddings (the paper derives K from the internet corpus
    # D^I), and the very same matrices are the retrieval index step 2 scores
    # against.  Embedding twice would let the two disagree about what a corpus
    # frame looks like, and it doubles the only expensive part of startup.
    #
    # K must exist before the first inference step because runner.run() calls
    # classify() on every 4th frame; the first bootstrap only runs after a whole
    # round of inference has finished, which is too late.
    index = _corpus_embeddings(args.corpus_index, args.corpus, embedder)
    kdm = _key_moment_model(args, embedder, index)

    def runner(**kw):
        envs = [_make_env(args.backend) for _ in range(args.num_agents)]
        try:
            r = InferenceRunner(
                lambda: envs[0], policy, kdm,
                action_space=action_space,
                image_size=args.image_size, device=device,
                max_steps=args.max_steps,
                # One action advances the agent's timestep, which is the same
                # interval the corpus is sampled at.  Leaving this at 1 frame
                # made the IDM learn 1/60 s dynamics and apply them to frames
                # 2 s apart.
                frame_skip=game.control_frame_skip,
            )
            return r.run(
                envs,
                delta=kw["delta"],
                timeout_s=kw["timeout_s"],
                random_steps=kw.get("random_steps", 0),
                random_seed=kw.get("random_seed", 0),
            )
        finally:
            for e in envs:
                e.close()

    def bootstrap_fn(**kw):  # full bootstrap; corpus loading lives here
        from ash.loop.bootstrap import Bootstrapper, BootstrapConfig

        b = Bootstrapper(BootstrapConfig(image_size=args.image_size, device=device))
        return b.run(
            kw["policy"], kw["idm"], kw["kdm"], kw["trajectories"],
            corpus_loader=_corpus_loader(args.corpus),
            out_dir=Path(kw["out_dir"]),
            # D^R from step 2.  Dropping it here made retrieval a no-op: the
            # bootstrap read the whole corpus and reported a plausible
            # per-round result while the retrieved set changed nothing.
            retrieved_ids=kw.get("retrieved_ids"),
            # Random-policy samples for IDM dynamics coverage (paper Alg 4).
            random_trajectories=kw.get("random_trajectories"),
            # The retrieval index already holds every corpus video's DINOv2
            # matrix; without it the bootstrap re-embeds D^R twice per round.
            corpus_embeddings=dict(index),
        )

    log.info("retrieval index: %d videos", len(index))

    orch = Orchestrator(
        policy, idm, kdm, embedder, index,
        config=LoopConfig(
            delta=args.delta,
            num_agents=args.num_agents,
            max_bootstraps=args.max_bootstraps,
            random_steps=args.random_steps,
            out_dir=out_dir,
        ),
        runner=runner, bootstrap_fn=bootstrap_fn,
    )
    result = orch.run()
    (out_dir / "loop-report.json").write_text(json.dumps(result, indent=2, default=str))
    log.info("done: %d bootstraps", result["bootstraps"])
    if args.backend == "cdp":
        log.info(
            "the game is deliberately left PAUSED (ticker stopped) so an unattended "
            "character is not beaten to death; resume with __ash.pump.resume() in the "
            "page, close(resume=True), or just restart the game"
        )
    return 0


def _is_menu_scene(env) -> bool:
    """True when the env reports a known menu template (agent.js V.MENU_SCENES)."""
    try:
        return bool((env.safety() or {}).get("menu"))
    except Exception:  # noqa: BLE001 - an unreadable scene must abort, not escape
        return False


def _policy_config(args, num_actions: int):
    """Policy config whose head width is the env's action space size.

    `num_actions` is a required argument, not a default: the head width must
    come from the environment, and an optional parameter would let a call site
    silently fall back to DEFAULT_NUM_ACTIONS while the env disagrees.
    """
    from ash.models.ash_policy import AshPolicyConfig

    return AshPolicyConfig(image_size=args.image_size, num_actions=num_actions)


def _idm_config(args, num_actions: int):
    """IDM config whose head width matches the policy's action space."""
    from ash.models.idm import IdmConfig

    return IdmConfig(image_size=args.image_size, num_actions=num_actions)


def _key_moment_model(args, embedder: Any, index: list[tuple[str, np.ndarray]]):
    """K, fit on the internet corpus D^I - reused from a cache when possible.

    Fitting is deterministic given the corpus and the hyperparameters, so a
    cached fit is equivalent to a fresh one (the tests pin that a reloaded
    model answers identically).  It is cached because it is minutes of CPU that
    every run otherwise pays for before the game is even touched: 113 s for the
    six-video corpus after the PCA reduction.

    K must exist before the first inference step because runner.run() calls
    classify() on every 4th frame; the first bootstrap only runs after a whole
    round of inference has finished, which is too late.
    """
    from ash.memory.kdm import KeyMomentModel

    kdm = KeyMomentModel()
    kdm.embedder = embedder
    if not index:
        log.warning(
            "no corpus under %s: key-moment discovery is unfitted, so every "
            "frame counts as noise and the agent will report stuck immediately. "
            "Run scripts/build_corpus.py first for a real run.",
            args.corpus,
        )
        return kdm

    cache = _kdm_cache_path(args)
    meta = _kdm_fingerprint(args, kdm, embedder)
    cached = KeyMomentModel.load(cache, meta)
    if cached is not None:
        cached.embedder = embedder
        log.info("loaded key-moment model from %s (%d kept clusters)",
                 cache, len(cached._kept))
        return cached

    n_frames = sum(len(e) for _, e in index)
    log.info("fitting key-moment model on %d frames from %d corpus videos",
             n_frames, len(index))
    report = kdm.fit(
        np.concatenate([e for _, e in index]),
        [vid for vid, e in index for _ in range(len(e))],
    )
    log.info("key-moment model: %s", report)
    kdm.save(cache, meta)
    log.info("wrote key-moment model to %s", cache)
    return kdm


def _kdm_cache_path(args) -> Path:
    """Sibling of the corpus dir, for the same reason as the embedding cache:
    the loader globs `*.npz` inside the corpus directory, so nothing the run
    produces may be written there."""
    if getattr(args, "kdm_cache", None):
        return Path(args.kdm_cache)
    base = Path(args.corpus or "data/corpus")
    return base.parent / (base.name + "-kdm.pkl")


def _kdm_fingerprint(args, kdm: Any, embedder: Any) -> str:
    """Everything a cached K depends on.

    Includes the clusterer libraries' versions: the pickle stores a fitted
    HDBSCAN, and approximate_predict is the code that runs at every inference
    step, so a version bump must invalidate the file rather than be trusted.
    """
    import sklearn

    return json.dumps(
        {
            "v": 1,
            "corpus": _corpus_fingerprint(args.corpus, embedder.image_size),
            "params": kdm.hyperparameters(),
            "hdbscan": _dist_version("hdbscan"),
            "sklearn": getattr(sklearn, "__version__", _dist_version("scikit-learn")),
        },
        sort_keys=True,
    )


def _dist_version(name: str) -> str:
    """hdbscan exposes no __version__, so read the installed metadata."""
    import importlib.metadata as md

    try:
        return md.version(name)
    except Exception:  # noqa: BLE001 - a missing version must not break a run
        return "unknown"


def _corpus_embeddings(
    path: str | None,
    corpus_dir: str | None,
    embedder: Any,
) -> list[tuple[str, np.ndarray]]:
    """One L2-normalized embedding matrix per corpus video.

    The same matrices serve both consumers - K is fit on them and step 2 scores
    against them - so this is built once, not once per consumer.

    Building it costs one DINOv2 pass over the corpus, so the result is cached
    in a sibling file.  The cache deliberately does NOT live inside the corpus
    directory: the loader globs `*.npz` there, so a cache written next to the
    videos would be picked up on the next run as a corpus video named "index".
    """
    base = Path(corpus_dir or "data/corpus")
    cache = Path(path) if path else base.parent / (base.name + "-embeddings.npz")
    stamp = _corpus_fingerprint(corpus_dir, embedder.image_size)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            meta = data["__meta__"] if "__meta__" in data.files else None
            if meta is not None and str(meta) == stamp:
                videos = [(k, data[k]) for k in data.files if k != "__meta__"]
                log.info("loaded retrieval index from %s (%d videos)", cache, len(videos))
                return videos
            # A stale index does not fail loudly: retrieval just scores against
            # the wrong frames.  Say why it is being rebuilt instead of silently
            # reusing it.
            log.warning("retrieval index cache %s is stale or unversioned; rebuilding", cache)
    out: list[tuple[str, np.ndarray]] = []
    for vid, frames in _corpus_loader(corpus_dir)():
        if len(frames):
            out.append((vid, embedder.embed(frames)))
    if out:
        np.savez_compressed(cache, __meta__=np.array(stamp), **dict(out))
        log.info("wrote retrieval index to %s (%d videos)", cache, len(out))
    return out


def _corpus_fingerprint(corpus_dir: str | None, image_size: int) -> str:
    """Cheap identity of the corpus + embedding resolution, for the cache.

    Uses file names and sizes rather than loading the frames: the point is to
    notice that the corpus changed, and reading 3 GB of npz to find that out
    would cost more than the embeddings it guards.
    """
    base = Path(corpus_dir or "data/corpus")
    files = sorted((p.name, p.stat().st_size) for p in base.glob("*.npz"))
    return json.dumps({"v": 1, "image_size": int(image_size), "files": files},
                      sort_keys=True)


def _corpus_loader(corpus_dir: str | None):
    """Yield (video_id, frames) for the corpus, optionally restricted to D^R.

    `ids=None` means the whole corpus D^I (used to fit K and to build the
    retrieval index).  `ids=[...]` means D^R only: the paper refits K and pi on
    the *retrieved* subset, so the bootstrap passes the ids step 2 selected.
    Reading the whole corpus regardless made retrieval a no-op.
    """
    def load(ids: list[str] | None = None):
        base = Path(corpus_dir or "data/corpus")
        want = None if ids is None else set(ids)
        for npz in sorted(base.glob("*.npz")):
            if want is not None and npz.stem not in want:
                continue
            data = np.load(npz)
            yield npz.stem, data["frames"]
    return load


def cmd_eval(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.report).read_text())
    print(json.dumps(data, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ash", description=__doc__)
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="environment self-check")
    d.add_argument("--backend", default="fake", choices=["fake", "cdp"])
    d.set_defaults(fn=cmd_doctor)

    r = sub.add_parser("run", help="self-hone loop: infer -> retrieve -> bootstrap")
    r.add_argument("--backend", default="fake", choices=["fake", "cdp"])
    r.add_argument("--policy", default=None, help="existing policy checkpoint")
    r.add_argument("--idm", default=None, help="existing IDM checkpoint")
    r.add_argument("--corpus", default="data/corpus", help="internet video corpus dir")
    r.add_argument("--corpus-index", default=None, help="precomputed embedding index json")
    r.add_argument("--kdm-cache", default=None,
                   help="fitted key-moment model cache (default: beside the corpus)")
    r.add_argument("--delta", type=int, default=600, help="stuck threshold in steps")
    r.add_argument("--num-agents", type=int, default=1)
    r.add_argument("--image-size", type=int, default=128)
    r.add_argument("--device", default=None)
    r.add_argument("--max-steps", type=int, default=20_000)
    #: The self-hone loop has no natural termination (the paper runs until the
    #: agent clears the game), so a dry run must be able to bound the number of
    #: rounds from the command line.  Without this the default of 64 bootstraps
    #: makes a smoke test indistinguishable from a hang.
    r.add_argument("--max-bootstraps", type=int, default=64)
    #: Random-policy steps per round, added to the IDM's training set (paper
    #: Alg 4 step 2).  0 disables the supplement and reproduces the old,
    #: narrow IDM training set.
    r.add_argument("--random-steps", type=int, default=100)
    r.add_argument("--out", default="runs/ash")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="print a loop/eval report")
    e.add_argument("report")
    e.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
