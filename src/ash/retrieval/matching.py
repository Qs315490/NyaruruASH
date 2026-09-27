"""Retrieval scoring (paper Algorithm 3).

Given a trajectory embedding matrix Q (q x d) and a corpus video embedding
matrix E (e x d), both L2-normalized, the similarity matrix is S = Q E^T.
Each candidate window of w_r consecutive corpus frames is scored by *greedy
one-to-one matching*: every corpus frame claims its highest-similarity
trajectory frame, and each trajectory frame is consumed at most once.  A
video's score is its best window score; top-k videos are retrieved per
trajectory and unioned into D^R.

The greedy loop in the paper is O(w_r^2) per window; the vectorized
equivalent here runs a full argmax over the window's similarity submatrix and
masks consumed columns in one pass, which is what the unit test in
tests/test_matching.py exercises against a brute-force reference.
"""

from __future__ import annotations

import numpy as np


def window_score(S: np.ndarray, start: int, w_r: int) -> float:
    """Greedy one-to-one match score of S[:, start:start+w_r].

    S is (q, e): rows are trajectory frames, columns are corpus frames.
    For each of the w_r corpus columns (in the window's own order), pick the
    highest-similarity remaining trajectory row, add its score, and remove
    that row from consideration.
    """
    W = S[:, start : start + w_r]
    available = np.ones(W.shape[0], dtype=bool)
    total = 0.0
    for j in range(W.shape[1]):
        masked = np.where(available, W[:, j], -np.inf)
        r = int(np.argmax(masked))
        if not np.isfinite(masked[r]):
            continue  # no trajectory frame left to match this column
        total += float(W[r, j])
        available[r] = False
    return total


def best_window_score(S: np.ndarray, w_r: int) -> float:
    """Best greedy window score over all temporal positions of one video."""
    q, e = S.shape
    if e == 0 or q == 0:
        return -np.inf
    w = min(w_r, e)
    best = -np.inf
    for start in range(0, e - w + 1):
        best = max(best, window_score(S, start, w))
    return best


def retrieve(
    query: np.ndarray,
    corpus: list[tuple[str, np.ndarray]],
    w_r: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Score every corpus video against one trajectory.

    query is (q, d) L2-normalized; corpus is a list of (video_id, E) with E
    (e, d) L2-normalized.  Returns the top_k (video_id, score) pairs, best
    first.  Scores are window sums of cosine similarities, so higher is more
    similar; -inf means the video had no frames to match.
    """
    scores: list[tuple[str, float]] = []
    for vid, E in corpus:
        S = query @ E.T
        scores.append((vid, best_window_score(S, w_r)))
    scores = [s for s in scores if np.isfinite(s[1])]
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[:top_k]
