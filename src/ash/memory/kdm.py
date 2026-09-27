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

import numpy as np

try:
    import hdbscan
except ImportError:  # pragma: no cover - optional dependency at import time
    hdbscan = None


class KeyMomentModel:
    def __init__(
        self,
        min_cluster_size: int = 12,
        min_distinct_trajectories: int = 3,
        metric: str = "euclidean",
    ) -> None:
        if hdbscan is None:
            raise ImportError("pip install hdbscan to use key-moment discovery")
        self.min_cluster_size = min_cluster_size
        self.min_distinct_trajectories = min_distinct_trajectories
        self.metric = metric
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
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.metric,
            approx_min_span_tree=True,
            prediction_data=True,
        )
        self._labels = self._clusterer.fit_predict(embeddings)
        self._fit_embeddings = embeddings
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

    # ---------- online classification ----------

    def cluster_of(self, embedding: np.ndarray) -> int:
        """Approximate-predict cluster id for one embedding (-1 = noise)."""
        if self._clusterer is None:
            raise RuntimeError("call fit() before classify()")
        label, _ = hdbscan.approximate_predict(self._clusterer, embedding.reshape(1, -1))
        return int(label[0])

    def classify(self, embedding: np.ndarray, seen_clusters: set[int]) -> bool:
        """K(obs, seen): True iff obs lands in a kept cluster not yet in seen."""
        if self._clusterer is None:
            raise RuntimeError("call fit() before classify()")
        label = self.cluster_of(embedding)
        return label >= 0 and label in self._kept and label not in seen_clusters
