"""Key-moment discovery and classification.

Pins the failure modes that made earlier progress-signal systems silently
wrong: a classifier that keeps firing on the same event would never let the
stuck timer advance, and a filter that keeps one-off clusters would promote
noise into progress.
"""

import numpy as np
import pytest

hdbscan = pytest.importorskip("hdbscan")

from ash.memory.kdm import KeyMomentModel  # noqa: E402


def make_blobs(n_clusters=3, per=20, d=8, seed=0):
    rng = np.random.default_rng(seed)
    embs, ids = [], []
    for c in range(n_clusters):
        center = rng.normal(size=d) * 3
        embs.append(center + rng.normal(size=(per, d)) * 0.1)
        # each cluster spans two videos so the distinct-trajectory filter passes
        ids.extend([f"video{c}a"] * (per // 2) + [f"video{c}b"] * (per - per // 2))
    return np.concatenate(embs).astype(np.float32), ids


def test_fit_keeps_multi_trajectory_clusters_only():
    embs, ids = make_blobs(n_clusters=3, per=20)
    # add a cluster that lives in a single video only -> must be dropped
    rng = np.random.default_rng(1)
    extra = rng.normal(size=(20, 8)).astype(np.float32) * 3 + 50
    embs = np.concatenate([embs, extra])
    ids = ids + ["solo"] * 20
    kdm = KeyMomentModel(min_cluster_size=5, min_distinct_trajectories=2)
    report = kdm.fit(embs, ids)
    assert report["clusters_kept"] == 3
    assert report["clusters_total"] >= 4


def test_classify_new_then_seen():
    embs, ids = make_blobs(n_clusters=2, per=30)
    kdm = KeyMomentModel(min_cluster_size=5, min_distinct_trajectories=2)
    kdm.fit(embs, ids)
    probe = embs[0]  # inside cluster 0
    seen: set[int] = set()
    assert kdm.classify(probe, seen) is True
    label = kdm.cluster_of(probe)
    assert label >= 0
    seen.add(label)
    # same event again must NOT count as a new key moment
    assert kdm.classify(probe, seen) is False


def test_unfit_raises():
    kdm = KeyMomentModel()
    with pytest.raises(RuntimeError):
        kdm.classify(np.zeros(8, dtype=np.float32), set())
