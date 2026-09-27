"""ASH self-hone loop (paper Algorithm 1).

    Step 1 Inference     N agents play under the shared policy pi until one
                         sees no new key moment for Delta steps (stuck).
    Step 2 Retrieval     embed the stuck trajectories with DINOv2, retrieve the
                         top-k visually similar videos from the corpus D^I.
    Step 3 Bootstrapping update K on D^R, update the IDM on the agents' own
                         trajectories, label D^R with IDM pseudo-actions and
                         update pi on the result.  Then go to step 1.

The orchestrator owns no environment details: it consumes an
``AgentRunner``-style callable for step 1 (see runner.py) and dataset objects
for step 3 (see bootstrap.py).  This module is the wiring, the paper's
"GoTo Step 1" is the loop, and every stopping condition is explicit.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ash.memory.kdm import KeyMomentModel
from ash.models.ash_policy import AshPolicy, load_policy
from ash.models.idm import IdmModel, load_idm
from ash.retrieval.matching import retrieve

log = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    #: stuck threshold: steps without a new key moment before bootstrapping
    delta: int = 600
    #: number of parallel agents (environments)
    num_agents: int = 1
    #: retrieval window and top-k
    w_r: int = 8
    top_k: int = 4
    #: wall-clock safety cap for one inference round (seconds)
    round_timeout_s: float = 8 * 3600
    #: max bootstraps before giving up (safety for unattended runs)
    max_bootstraps: int = 64
    #: random-policy steps per round, added to the IDM's training set.  The
    #: paper supplements the agent's own transitions this way; without it the
    #: IDM only sees the narrow slice of dynamics the current policy produces.
    random_steps: int = 100
    #: where checkpoints and reports are written
    out_dir: Path = field(default_factory=lambda: Path("runs/ash"))


class Orchestrator:
    """Drive the infer -> retrieve -> bootstrap cycle."""

    def __init__(
        self,
        policy: AshPolicy,
        idm: IdmModel,
        kdm: KeyMomentModel,
        embedder: Any,
        corpus_index: list[tuple[str, np.ndarray]],
        *,
        config: LoopConfig,
        # runner(step_fn, agents) -> per-agent trajectory dicts; provided by
        # the CLI so the loop itself stays environment-agnostic.
        runner: Callable[..., Any],
        bootstrap_fn: Callable[..., Any],
    ) -> None:
        self.policy = policy
        self.idm = idm
        self.kdm = kdm
        self.embedder = embedder
        self.corpus_index = corpus_index
        self.config = config
        self.runner = runner
        self.bootstrap_fn = bootstrap_fn
        self.bootstrap_round = 0
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """Run the full self-hone cycle until a stopping condition fires.

        The report is written to disk after every stage instead of only at the
        end.  A run that dies in the bootstrap - the long part - used to leave
        nothing at all, so the round's inference statistics were lost with it
        and there was no way to tell whether the round had seen a single key
        moment.  `bootstrap_pending` marks a round whose bootstrap never
        finished.
        """
        report_doc: dict[str, Any] = {"rounds": [], "bootstraps": 0, "done": False}
        history: list[dict] = report_doc["rounds"]
        while self.bootstrap_round < self.config.max_bootstraps:
            started = time.time()
            result = self.runner(
                policy=self.policy,
                kdm=self.kdm,
                delta=self.config.delta,
                timeout_s=self.config.round_timeout_s,
                random_steps=self.config.random_steps,
                # Different exploration every round: a fixed seed replayed the
                # same hundred actions forever and never widened coverage.
                random_seed=self.bootstrap_round,
            )
            trajectories = result["trajectories"]          # list of {"obs","act"}
            random_trajectories = result.get("random_trajectories", [])
            seen_stats = {
                "steps": result.get("steps"),
                "stuck": result.get("stuck"),
                "key_moments": [len(m) for m in result.get("memories", [])],
                "aborted": result.get("aborted"),
                # K's verdicts, so a round with no key moment says which kind of
                # nothing it was: all noise, or all one already-seen cluster.
                "key_stats": result.get("key_stats"),
                "maps": result.get("maps"),
                # How close the round's own frames got to the corpus.  Corpus
                # frames sit ~0.93 cosine from their nearest corpus frame; a live
                # view far below that is out of distribution, and no threshold or
                # refit will make K fire on it.
                "best_corpus_cosine": self._best_corpus_cosine(trajectories),
            }
            log.info(
                "inference round %d: %d agents, stats=%s (%.0fs)",
                self.bootstrap_round,
                len(trajectories),
                seen_stats,
                time.time() - started,
            )

            # Step 2: retrieval per agent trajectory, union into D^R.
            retrieved: list[tuple[str, float]] = []
            for traj in trajectories:
                # A trajectory is {"obs": (T,H,W,C), "act": (T,)}; retrieval only
                # needs the frames.
                emb = self.embedder.embed(np.asarray(traj["obs"]))
                retrieved.extend(retrieve(emb, self.corpus_index, self.config.w_r, self.config.top_k))
            vid_rank: dict[str, int] = {}
            for vid, score in retrieved:
                vid_rank.setdefault(vid, len(vid_rank))
            d_r = [vid for vid, _ in sorted(vid_rank.items(), key=lambda kv: kv[1])]
            log.info("retrieved %d videos for bootstrap: %s", len(d_r), d_r)

            entry: dict[str, Any] = {
                "round": self.bootstrap_round,
                "retrieved": d_r,
                "stats": seen_stats,
                "bootstrap_pending": True,
            }
            history.append(entry)
            self._save_round_frames(entry["round"], trajectories)
            self.write_report(report_doc)

            # Step 3: bootstrap K, IDM, pi.
            entry.update(self.bootstrap_fn(
                policy=self.policy,
                idm=self.idm,
                kdm=self.kdm,
                trajectories=trajectories,
                random_trajectories=random_trajectories,
                retrieved_ids=d_r,
                # Seeds the demonstration replay sample: a different slice each
                # round, so the whole recording is eventually replayed.
                round_index=self.bootstrap_round,
                out_dir=self.config.out_dir / f"bootstrap-{self.bootstrap_round:03d}",
            ))
            entry.pop("bootstrap_pending", None)
            self.bootstrap_round += 1
            report_doc["bootstraps"] = self.bootstrap_round
            self.write_report(report_doc)
        report_doc["done"] = True
        self.write_report(report_doc)
        return report_doc

    @staticmethod
    def _sample_frames(trajectory: dict, limit: int = 64) -> np.ndarray:
        """Up to `limit` frames spread across a round.  Kept small on purpose:
        this is for looking at what the round saw, not for training."""
        obs = np.asarray(trajectory["obs"])
        if not len(obs):
            return obs
        step = max(1, len(obs) // limit)
        return obs[::step][:limit]

    def _save_round_frames(self, round_index: int, trajectories: list) -> None:
        """Write a sample of the round's frames next to its report.

        Without this there is no way to answer "why did K see nothing?" after
        the fact: the embeddings are gone, the game has moved on, and a later
        capture of a paused game is a frozen frame.  That dead end cost three
        separate attempts before this existed.
        """
        if not trajectories:
            return
        frames = self._sample_frames(trajectories[0])
        if not len(frames):
            return
        path = self.config.out_dir / f"round-{round_index:03d}-frames.npz"
        try:
            np.savez_compressed(path, frames=frames)
        except OSError as exc:  # a diagnostic must never fail the run
            log.warning("could not write %s: %s", path, exc)

    def _best_corpus_cosine(self, trajectories: list) -> float | None:
        """Best cosine between the round's own frames and any corpus frame.

        Sampled, not exhaustive: the answer is a distribution check, and the
        full product over 68440 corpus frames per trajectory is a second copy of
        the retrieval pass for a number that moves by ~1e-3.
        """
        if not self.corpus_index or not trajectories:
            return None
        obs = np.asarray(trajectories[0]["obs"])
        if not len(obs):
            return None
        step = max(1, len(obs) // 32)
        emb = self.embedder.embed(obs[::step])
        return max(float((emb @ mat.T).max()) for _, mat in self.corpus_index)

    def write_report(self, document: dict) -> None:
        """Persist the report, atomically so a kill cannot leave half a file."""
        path = self.config.out_dir / "loop-report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(document, indent=2, default=str))
        tmp.replace(path)


def load_or_init_policy(path: str | None, config: Any) -> AshPolicy:
    return load_policy(path) if path else AshPolicy(config)


def load_or_init_idm(path: str | None, config: Any) -> IdmModel:
    return load_idm(path) if path else IdmModel(config)
