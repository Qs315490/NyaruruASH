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


def test_pca_projection_is_shared_between_fit_and_online_queries():
    """fit() and classify() must project identically.

    Fitting on PCA-reduced features while asking the clusterer about raw 384-d
    embeddings does not raise: approximate_predict just returns noise, so every
    frame would silently stop being a key moment.
    """
    import numpy as np

    from ash.memory.kdm import KeyMomentModel

    rng = np.random.RandomState(0)
    # Three well-separated blobs repeated across ids, in a wide space.
    centres = rng.randn(3, 384) * 6
    rows, ids = [], []
    for vid in range(4):
        for c in range(3):
            rows.append(centres[c] + rng.randn(40, 384) * 0.2)
            ids += [vid] * 40
    X = np.concatenate(rows).astype("float32")

    m = KeyMomentModel(min_cluster_size=10, min_distinct_trajectories=2, pca_dim=8)
    m.fit(X, np.asarray(ids))

    assert m._pca is not None
    assert m._project(X[:5]).shape == (5, 8), "fit-space is the PCA space"
    # A raw embedding from a kept cluster must be recognised, not rejected.
    labels = {m.cluster_of(centres[c] + rng.randn(384) * 0.05) for c in range(3)}
    assert any(l >= 0 for l in labels), (
        "queries must be projected the same way as the fit"
    )


def test_pca_can_be_disabled():
    import numpy as np

    from ash.memory.kdm import KeyMomentModel

    X = np.random.RandomState(0).randn(60, 32).astype("float32")
    m = KeyMomentModel(min_cluster_size=5, pca_dim=None)
    m.fit(X, np.zeros(60, dtype=int))
    assert m._pca is None
    assert m._project(X[:3]).shape == (3, 32)


# --------------------------------------------------------------------------
# Persistence.  Fitting K is minutes of CPU (113 s for the six-video corpus),
# so it is cached; a cache that loads when it should not is worse than no
# cache at all, because HDBSCAN answers a mismatched query with "noise".
# --------------------------------------------------------------------------


def _fitted(pca_dim=8, d=32):
    rng = np.random.RandomState(0)
    centres = rng.randn(3, d) * 6
    rows, ids = [], []
    for vid in range(4):
        for c in range(3):
            rows.append(centres[c] + rng.randn(30, d) * 0.2)
            ids += [f"v{vid}"] * 30
    X = np.concatenate(rows).astype("float32")
    m = KeyMomentModel(min_cluster_size=10, min_distinct_trajectories=2, pca_dim=pca_dim)
    m.fit(X, np.asarray(ids))
    return m, X


def test_saved_model_answers_identically(tmp_path):
    m, X = _fitted()
    path = tmp_path / "k.pkl"
    m.save(path, meta="same")

    back = KeyMomentModel.load(path, meta="same")
    assert back is not None
    assert m._kept == back._kept
    assert m._pca is not None and back._pca is not None
    assert np.array_equal(m._labels, back._labels)
    # The projection must survive too, or every online query becomes noise.
    assert np.array_equal(m._project(X[:7]), back._project(X[:7]))
    assert [m.cluster_of(X[i]) for i in range(9)] == [back.cluster_of(X[i]) for i in range(9)]
    probe = X[0]
    assert back.classify(probe, set()) == m.classify(probe, set())


def test_load_returns_none_when_absent(tmp_path):
    assert KeyMomentModel.load(tmp_path / "missing.pkl", meta="x") is None


def test_load_rejects_a_changed_fingerprint(tmp_path):
    m, _ = _fitted()
    path = tmp_path / "k.pkl"
    m.save(path, meta="corpus-a")
    assert KeyMomentModel.load(path, meta="corpus-b") is None


def test_load_rebuilds_on_a_corrupt_or_foreign_file(tmp_path):
    garbage = tmp_path / "garbage.pkl"
    garbage.write_bytes(b"not a pickle at all")
    assert KeyMomentModel.load(garbage, meta="x") is None

    import pickle

    foreign = tmp_path / "foreign.pkl"
    foreign.write_bytes(pickle.dumps({"meta": "x"}))
    assert KeyMomentModel.load(foreign, meta="x") is None

    wrong_arity = tmp_path / "arity.pkl"
    wrong_arity.write_bytes(pickle.dumps(("x", None, None)))
    assert KeyMomentModel.load(wrong_arity, meta="x") is None


def test_cache_does_not_carry_the_embedder(tmp_path):
    """The embedder is attached by the caller and must not be pickled: a torch
    module would bloat the file and tie it to a torch version."""
    m, X = _fitted()
    m.embedder = lambda frames: np.zeros((len(frames), 32), dtype=np.float32)
    path = tmp_path / "k.pkl"
    m.save(path, meta="x")  # an unpicklable embedder must not break the save

    back = KeyMomentModel.load(path, meta="x")
    assert back is not None
    assert not hasattr(back, "embedder")


def test_saving_an_unfit_model_is_refused(tmp_path):
    with pytest.raises(RuntimeError):
        KeyMomentModel(pca_dim=8).save(tmp_path / "k.pkl", meta="x")


def test_cache_version_mismatch_is_rebuilt(tmp_path):
    import pickle

    m, _ = _fitted()
    path = tmp_path / "k.pkl"
    m.save(path, meta="x")
    meta, model = pickle.loads(path.read_bytes())
    model.CACHE_VERSION = KeyMomentModel.CACHE_VERSION + 1
    path.write_bytes(pickle.dumps((meta, model)))
    assert KeyMomentModel.load(path, meta="x") is None


def test_classify_sequence_matches_the_frame_by_frame_loop():
    """The batched query must be exactly the per-frame definition, not an
    approximation of it: the two are used interchangeably (bootstrap uses the
    batched one, the live runner the per-frame one)."""
    m, X = _fitted()
    flags = m.classify_sequence(X)
    seen: set[int] = set()
    want = np.zeros(len(X), dtype=bool)
    for t in range(len(X)):
        if m.classify(X[t], seen):
            want[t] = True
            seen.add(m.cluster_of(X[t]))
    assert np.array_equal(flags, want)
    assert flags.any(), "the fixture must contain at least one key moment"

    # A second pass over the same trajectory sees the same events as "seen".
    assert not m.classify_sequence(X).sum() == 0


def test_classify_sequence_edge_shapes():
    m, X = _fitted()
    assert m.classify_sequence(X[:0]).shape == (0,)
    assert m.classify_sequence(X[0]).shape == (1,)
    with pytest.raises(RuntimeError):
        KeyMomentModel(pca_dim=8).classify_sequence(X[:1])


def test_observe_matches_classify_and_reports_why_it_did_not_fire():
    """classify() is observe()[0], and the verdict must distinguish the two
    kinds of "no key moment": noise, versus a cluster already seen."""
    m, X = _fitted()
    seen: set[int] = set()
    for i in range(len(X)):
        fired = m.classify(X[i], seen)
        replayed, label, is_key = m.observe(X[i], seen)
        assert replayed == fired, i
        assert is_key == (label >= 0 and label in m._kept), i
        if fired:
            seen.add(label)

    # A frame from a kept cluster that the trajectory has already matched is
    # "no key moment" for a reason that is NOT noise.
    keeper = next(i for i in range(len(X)) if m.observe(X[i], set())[2])
    label = m.cluster_of(X[keeper])
    assert m.observe(X[keeper], set())[0] is True
    again = m.observe(X[keeper], {label})
    assert again[0] is False and again[1] == label and again[2] is True


def test_observe_counts_are_what_the_runner_reports():
    """The runner's counters must come from observe(), not from re-deriving it."""
    from ash.loop.runner import AgentState

    s = AgentState(agent_id=0)
    assert (s.k_evals, s.k_noise, s.k_in_key_cluster, s.k_fired) == (0, 0, 0, 0)
    m, X = _fitted()
    # noise: nothing in the fixture is noise, so use a far-away vector
    _, label, is_key = m.observe(np.full(X.shape[1], 1e3, dtype=np.float32), set())
    assert label == -1 and is_key is False
