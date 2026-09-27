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
        """Run the full self-hone cycle until a stopping condition fires."""
        history: list[dict] = []
        while self.bootstrap_round < self.config.max_bootstraps:
            started = time.time()
            result = self.runner(
                policy=self.policy,
                kdm=self.kdm,
                delta=self.config.delta,
                timeout_s=self.config.round_timeout_s,
            )
            trajectories = result["trajectories"]          # list of obs arrays
            seen_stats = result.get("stats", {})
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
                emb = self.embedder.embed(np.asarray(traj))
                retrieved.extend(retrieve(emb, self.corpus_index, self.config.w_r, self.config.top_k))
            vid_rank: dict[str, int] = {}
            for vid, score in retrieved:
                vid_rank.setdefault(vid, len(vid_rank))
            d_r = [vid for vid, _ in sorted(vid_rank.items(), key=lambda kv: kv[1])]
            log.info("retrieved %d videos for bootstrap: %s", len(d_r), d_r)

            # Step 3: bootstrap K, IDM, pi.
            report = self.bootstrap_fn(
                policy=self.policy,
                idm=self.idm,
                kdm=self.kdm,
                trajectories=trajectories,
                retrieved_ids=d_r,
                out_dir=self.config.out_dir / f"bootstrap-{self.bootstrap_round:03d}",
            )
            report["round"] = self.bootstrap_round
            report["retrieved"] = d_r
            report["stats"] = seen_stats
            history.append(report)
            self.bootstrap_round += 1
        return {"rounds": history, "bootstraps": self.bootstrap_round}


def load_or_init_policy(path: str | None, config: Any) -> AshPolicy:
    return load_policy(path) if path else AshPolicy(config)


def load_or_init_idm(path: str | None, config: Any) -> IdmModel:
    return load_idm(path) if path else IdmModel(config)
