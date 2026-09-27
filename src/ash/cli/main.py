"""ASH CLI: doctor, bootstrap-init, run, eval.

The commands mirror the ASH cycle:

    ash doctor        environment self-check (CDP, game, GPU, corpus)
    ash init-kdm      fit the key-moment model on an indexed corpus
    ash run           inference rounds with automatic bootstrapping
    ash eval          milestone-style progress report from a trajectory dump

Every command works with --backend fake for offline dry runs; the CDP backend
is only required for real play.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
        env = make_env(args.backend)
        obs = env.reset()
        checks.append((f"backend:{args.backend}", f"obs {obs.rgb.shape}"))
        env.close()
    except Exception as e:  # pragma: no cover - diagnostic path
        checks.append((f"backend:{args.backend}", f"FAIL: {e}"))
    for name, status in checks:
        print(f"  {name:<24} {status}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from ash.env.base import make_env
    from ash.loop.orchestrator import LoopConfig, Orchestrator, load_or_init_idm, load_or_init_policy
    from ash.loop.runner import InferenceRunner

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = load_or_init_policy(args.policy, _policy_config(args))
    idm = load_or_init_idm(args.idm, _idm_config(args))

    from ash.memory.embeddings import FrameEmbedder
    from ash.memory.kdm import KeyMomentModel

    embedder = FrameEmbedder()
    kdm = KeyMomentModel()
    kdm.embedder = embedder

    def runner(**kw):
        envs = [make_env(args.backend) for _ in range(args.num_agents)]
        try:
            r = InferenceRunner(
                lambda: envs[0], policy, kdm,
                image_size=args.image_size, device=args.device,
                max_steps=args.max_steps,
            )
            return r.run(envs, delta=kw["delta"], timeout_s=kw["timeout_s"])
        finally:
            for e in envs:
                e.close()

    def bootstrap_fn(**kw):  # full bootstrap; corpus loading lives here
        from ash.loop.bootstrap import Bootstrapper, BootstrapConfig

        b = Bootstrapper(BootstrapConfig(image_size=args.image_size, device=args.device))
        return b.run(
            kw["policy"], kw["idm"], kw["kdm"], kw["trajectories"],
            corpus_loader=_corpus_loader(args.corpus),
            out_dir=Path(kw["out_dir"]),
        )

    orch = Orchestrator(
        policy, idm, kdm, embedder, _corpus_index(args.corpus_index),
        config=LoopConfig(delta=args.delta, num_agents=args.num_agents, out_dir=out_dir),
        runner=runner, bootstrap_fn=bootstrap_fn,
    )
    result = orch.run()
    (out_dir / "loop-report.json").write_text(json.dumps(result, indent=2, default=str))
    log.info("done: %d bootstraps", result["bootstraps"])
    return 0


def _policy_config(args):
    from ash.models.ash_policy import AshPolicyConfig

    return AshPolicyConfig(image_size=args.image_size)


def _idm_config(args):
    from ash.models.idm import IdmConfig

    return IdmConfig(image_size=args.image_size)


def _corpus_index(path: str | None):
    if not path or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text())


def _corpus_loader(corpus_dir: str | None):
    def load(retrieved_only: bool = False):
        base = Path(corpus_dir or "data/corpus")
        for npz in sorted(base.glob("*.npz")):
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
    r.add_argument("--delta", type=int, default=600, help="stuck threshold in steps")
    r.add_argument("--num-agents", type=int, default=1)
    r.add_argument("--image-size", type=int, default=128)
    r.add_argument("--device", default=None)
    r.add_argument("--max-steps", type=int, default=20_000)
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
