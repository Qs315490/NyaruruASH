"""Greedy one-to-one matching must match the paper's loop exactly.

The paper scores a window by: for each corpus frame (in order), claim the
highest-similarity *remaining* trajectory frame; each trajectory frame is
consumed at most once.  A vectorized rewrite that just sums column maxima is
NOT equivalent - it double-counts trajectory frames - so this file pins the
implementation to a brute-force reference.
"""

import numpy as np
import pytest

from ash.retrieval.matching import best_window_score, retrieve, window_score


def brute_window(S: np.ndarray, start: int, w_r: int) -> float:
    W = S[:, start : start + w_r]
    used: set[int] = set()
    total = 0.0
    for j in range(W.shape[1]):
        best_r, best_v = -1, -np.inf
        for r in range(W.shape[0]):
            if r in used:
                continue
            if W[r, j] > best_v:
                best_v, best_r = W[r, j], r
        if best_r < 0:
            continue
        total += float(best_v)
        used.add(best_r)
    return total


def test_window_score_matches_bruteforce():
    rng = np.random.default_rng(7)
    for _ in range(50):
        q, e, w = rng.integers(1, 8, size=3)
        S = rng.uniform(-1, 1, size=(q, e))
        start = int(rng.integers(0, max(1, e - w + 1)))
        assert window_score(S, start, w) == pytest.approx(brute_window(S, start, w))


def test_one_to_one_no_double_count():
    # Two corpus frames both prefer trajectory row 0; the second must fall
    # back to row 1, and the total must reflect that, not 2 * W[0, :].
    S = np.array([[0.9, 0.8], [0.1, 0.7]])
    # window over both columns: col0 takes row0 (0.9), col1 takes row1 (0.7)
    assert window_score(S, 0, 2) == pytest.approx(1.6)


def test_best_window_picks_max_over_starts():
    S = np.array([[0.1, 0.9, 0.1], [0.1, 0.2, 0.9]])
    scores = [window_score(S, t, 1) for t in range(3)]
    assert best_window_score(S, 1) == max(scores)


def test_retrieve_orders_and_topk():
    q = np.eye(4)
    corpus = [
        ("exact", np.eye(4)),
        ("orthogonal", np.fliplr(np.eye(4)) * 0.0 + np.diag([1, 0, 0, 0])),
        ("noise", np.random.default_rng(0).normal(size=(4, 4)) * 0.1),
    ]
    got = retrieve(q, corpus, w_r=2, top_k=2)
    assert len(got) == 2
    assert got[0][0] == "exact"
    assert got[0][1] >= got[1][1]
