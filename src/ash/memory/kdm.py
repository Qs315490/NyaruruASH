"""Key-moment discovery and classification (ASH paper section 3, "Discovering key moments").

K is a binary classifier over observations: K(obs) is True when the observation
is a *new* key moment - its HDBSCAN cluster exists and no earlier observation in
the current trajectory has already matched that cluster.

Discovery: HDBSCAN clusters the DINOv2 embeddings of the retrieved internet
corpus (plus all previously clustered embeddings).  Clusters whose members come
from fewer than c_min distinct trajectories are dropped - they are noise or
one-off events, not recurring key moments.

Why HDBSCAN (paper's three reasons): the number of key moments is unknown,
outliers are rejected as noise instead of forced into a cluster, and
approximate_predict supports online assignment of new frames without a refit.

The state is a plain dict {cluster_id: bool-seen} per trajectory; persisting
that dict between episodes is the caller's concern.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional dependency at import time
    hdbscan = None

log = logging.getLogger(__name__)


class KeyMomentModel:
    #: Bumped whenever the pickled layout changes, so an old file is rebuilt
    #: instead of being unpickled into a differently shaped object.
    CACHE_VERSION = 1

    def __init__(
        self,
        min_cluster_size: int = 12,
        min_distinct_trajectories: int = 3,
        metric: str = "euclidean",
        pca_dim: int | None = 64,
    ) -> None:
        if hdbscan is None:
            raise ImportError("pip install hdbscan to use key-moment discovery")
        self.min_cluster_size = min_cluster_size
        self.min_distinct_trajectories = min_distinct_trajectories
        self.metric = metric
        #: Cluster in this many PCA components instead of the raw DINOv2 width.
        #: HDBSCAN's k-NN search in 384 dimensions did not finish in six minutes
        #: on the 6-video corpus (68440 frames); at 64 components - which keep
        #: 81% of the variance - the same fit took 112 s.  None disables it.
        self.pca_dim = pca_dim
        self._pca = None
        self._fit_embeddings: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._clusterer = None

    # ---------- discovery ----------

    def fit(self, embeddings: np.ndarray, trajectory_ids: np.ndarray | list) -> dict:
        """Cluster embeddings; keep clusters spanning >= min_distinct_trajectories.

        trajectory_ids parallel the embeddings and say which video each came
        from, so the distinct-trajectory filter can run.
        """
        if len(embeddings) != len(trajectory_ids):
            raise ValueError("embeddings and trajectory_ids must be parallel")
        ids = np.asarray(trajectory_ids)
        features = np.asarray(embeddings, dtype=np.float32)
        self._pca = None
        if self.pca_dim is not None and features.shape[1] > self.pca_dim:
            self._pca = PCA(n_components=self.pca_dim, random_state=0).fit(features)
        features = self._project(features)
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.metric,
            approx_min_span_tree=True,
            prediction_data=True,
            # Computing the core distances is the single most expensive step and
            # is embarrassingly parallel, but the library's default leaves most
            # cores idle: fitting the 6-video corpus (68440 frames) sat on one
            # core for many minutes with an idle GPU and idle CPU.  -1 uses all
            # of them.  This does not change the result, only the wall time.
            core_dist_n_jobs=-1,
        )
        self._labels = self._clusterer.fit_predict(features)
        self._fit_embeddings = features
        kept = set()
        for label in np.unique(self._labels):
            if label < 0:
                continue  # noise
            members = ids[self._labels == label]
            if len(np.unique(members)) >= self.min_distinct_trajectories:
                kept.add(int(label))
        # Relabel to a dense id space and remember which survive.
        self._kept = kept
        return {
            "clusters_total": int(len(np.unique(self._labels[self._labels >= 0]))),
            "clusters_kept": len(kept),
            "noise_rate": float((self._labels < 0).mean()),
        }

    def _project(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the fitted PCA.  Online queries MUST use the same projection as
        the fit, or the clusterer receives a differently shaped vector."""
        X = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if self._pca is None:
            return X
        return self._pca.transform(X).astype(np.float32)

    # ---------- online classification ----------

    def cluster_of(self, embedding: np.ndarray) -> int:
        """Approximate-predict cluster id for one embedding (-1 = noise)."""
        if self._clusterer is None:
            raise RuntimeError("call fit() before classify()")
        label, _ = hdbscan.approximate_predict(
            self._clusterer, self._project(embedding))
        return int(label[0])

    def observe(
        self, embedding: np.ndarray, seen_clusters: set[int]
    ) -> tuple[bool, int, bool]:
        """One online K evaluation: (new key moment?, cluster id, is a key cluster?).

        classify() is this without the diagnostics.  The runner needs the raw
        verdict because a round that reports no key moment has two opposite
        explanations: every evaluation landed in noise (the live view is outside
        the corpus' feature space) or they all landed in a cluster already seen
        (the agent spent the round looking at one thing).  `key_moments: 0` alone
        cannot tell them apart, and the fixes are different.
        """
        if self._clusterer is None:
            raise RuntimeError("call fit() before observe()")
        label = self.cluster_of(embedding)
        is_key = label >= 0 and label in self._kept
        return (is_key and label not in seen_clusters), label, is_key

    def classify(self, embedding: np.ndarray, seen_clusters: set[int]) -> bool:
        """K(obs, seen): True iff obs lands in a kept cluster not yet in seen."""
        return self.observe(embedding, seen_clusters)[0]

    def classify_sequence(self, embeddings: np.ndarray) -> np.ndarray:
        """K over a whole trajectory: (N,) bool, one flag per frame.

        Identical to calling classify() frame by frame with an accumulating
        `seen` set (a frame is a key moment when it is the first member of its
        kept cluster), but approximate_predict is asked once for all N frames.
        The per-frame version costs 3.4 ms each - measured - because every call
        redoes the PCA transform and re-queries the KD-tree, which is 3.9
        minutes of one core for a 68440-frame corpus pass.
        """
        if self._clusterer is None:
            raise RuntimeError("call fit() before classify_sequence()")
        embs = np.asarray(embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs[None]
        if len(embs) == 0:
            return np.zeros(0, dtype=bool)
        labels, _ = hdbscan.approximate_predict(self._clusterer, self._project(embs))
        flags = np.zeros(len(labels), dtype=bool)
        seen: set[int] = set()
        for t, label in enumerate(labels.tolist()):
            if label >= 0 and label in self._kept and label not in seen:
                flags[t] = True
                seen.add(label)
        return flags

    # ---------- persistence ----------

    #: The attributes that make up a *fitted* model.  Enumerating them, instead
    #: of subtracting from ``__dict__``, keeps the caller-attached embedder and
    #: any per-instance monkeypatch out of the file.
    _FITTED_FIELDS = (
        "min_cluster_size",
        "min_distinct_trajectories",
        "metric",
        "pca_dim",
        "_pca",
        "_labels",
        "_fit_embeddings",
        "_clusterer",
        "_kept",
    )

    def hyperparameters(self) -> dict:
        """The fit inputs that are not the corpus itself."""
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_distinct_trajectories": self.min_distinct_trajectories,
            "metric": self.metric,
            "pca_dim": self.pca_dim,
        }

    def __getstate__(self) -> dict:
        """Pickle the fitted model only.

        ``embedder`` is attached by the caller (runner and bootstrap read it off
        the model) and is not part of the fit: pickling a torch module would
        bloat the cache and tie its validity to a torch version.  A clusterer
        fitted *without* prediction data cannot answer online queries, so the
        unfitted/partial states are simply absent from the file.
        """
        state = {"cache_version": self.CACHE_VERSION}
        for name in self._FITTED_FIELDS:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state: dict) -> None:
        version = state.get("cache_version")
        if version != self.CACHE_VERSION:
            raise ValueError(
                f"key-moment cache version {version!r} != {self.CACHE_VERSION}")
        missing = [n for n in ("_clusterer", "_labels", "_kept") if n not in state]
        if missing:
            raise ValueError(f"key-moment cache is missing {missing}")
        self.__dict__.update({k: v for k, v in state.items() if k != "cache_version"})

    def save(self, path: str | Path, meta: str) -> Path:
        """Write the fitted model next to ``meta``; the write is atomic so a
        killed process cannot leave a truncated file that still looks valid.

        Re-saving an unfit model is refused: the file would load and then fail
        much later, at the first ``classify()``.
        """
        if self._clusterer is None:
            raise RuntimeError("call fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump((meta, self), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path, meta: str) -> KeyMomentModel | None:
        """Load a cached fit, or None when it is unusable.

        ``meta`` is an opaque fingerprint of everything the fit depends on
        (corpus contents, embedding resolution, hyperparameters, clusterer
        versions).  Loading across a mismatch would keep classifying frames
        against clusters that no longer describe the corpus, and because
        HDBSCAN answers unknown inputs with "noise" that failure would be
        silent.  Any unreadable or foreign file is reported and rebuilt rather
        than raising: a cache is an optimisation, never a source of truth.

        The caller must re-attach the embedder; it is deliberately not stored.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            if not isinstance(payload, tuple) or len(payload) != 2:
                raise ValueError("cache payload is not a (meta, model) pair")
            cached_meta, model = payload
            if not isinstance(cached_meta, str) or cached_meta != meta:
                log.warning(
                    "key-moment cache %s is stale (corpus, resolution or "
                    "hyperparameters changed); refitting", path)
                return None
            if not isinstance(model, cls):
                raise ValueError(f"cache holds a {type(model).__name__}")
            return model
        except Exception as exc:  # noqa: BLE001 - see docstring: rebuild, never fail
            log.warning("key-moment cache %s is unusable (%s: %s); refitting",
                        path, type(exc).__name__, exc)
            return None
