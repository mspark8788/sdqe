"""최적화된 거리 계산이 단순 구현(이중 루프)과 같은 결과를 내는지 검증합니다."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_mixed

from sdqe import config
from sdqe.common import distance as D
from sdqe.preprocess.validation import validate_pair


def naive_distance(a, b, i, j) -> float:
    """정의를 그대로 옮긴 한 쌍의 거리 (파이썬 루프)."""
    p = a.n_features
    if a.metric == "gower":
        s, cnt = 0.0, 0
        for k in range(a.numeric.shape[1]):
            x, y = a.numeric[i, k], b.numeric[j, k]
            if not (np.isnan(x) or np.isnan(y)):
                s += abs(x - y)
                cnt += 1
        mism = sum(
            int(a.categorical[i, c] != b.categorical[j, c]) for c in range(a.categorical.shape[1])
        )
        denom = cnt + a.categorical.shape[1]
        return round((s + mism) / denom, config.DISTANCE_DECIMALS) if denom else np.inf
    matches = sum(
        int(a.categorical[i, c] == b.categorical[j, c]) for c in range(a.categorical.shape[1])
    )
    cos = float(np.dot(a.numeric[i], b.numeric[j])) if a.numeric.shape[1] else 0.0
    sim = (2 * matches - a.categorical.shape[1] + a.numeric.shape[1] * cos) / p
    return round(1 - sim, config.DISTANCE_DECIMALS)


def naive_matrix(a, b):
    return np.array(
        [[naive_distance(a, b, i, j) for j in range(b.n_rows)] for i in range(a.n_rows)]
    )


@pytest.fixture(params=["gower", "cosine"])
def feats(request):
    data = validate_pair(make_mixed(40, 3), make_mixed(30, 4))
    return D.prepare_features(data.original, [data.synthetic], data.schema, request.param)


def test_block_matches_naive(feats):
    fo, fs = feats
    np.testing.assert_allclose(D.distance_block(fs, fo), naive_matrix(fs, fo), rtol=0, atol=1e-12)


def test_gower_scaling_uses_original_range():
    data = validate_pair(make_mixed(40, 3), make_mixed(30, 4))
    fo, _ = D.prepare_features(data.original, [data.synthetic], data.schema, "gower")
    col = data.schema.numeric[0]
    rng = data.original[col].max() - data.original[col].min()
    expected = (data.original[col] - data.original[col].min()) / rng
    np.testing.assert_allclose(fo.numeric[:, 0], expected)


def test_gower_missing_numeric_excluded_from_denominator():
    orig = make_mixed(20, 5)
    syn = make_mixed(15, 6).astype({"income": float})
    syn.loc[[0, 3], "income"] = np.nan
    data = validate_pair(orig, syn)
    fo, fs = D.prepare_features(data.original, [data.synthetic], data.schema, "gower")
    np.testing.assert_allclose(D.distance_block(fs, fo), naive_matrix(fs, fo), atol=1e-12)


def test_nearest_neighbor_ties_pick_lowest_index(feats, monkeypatch):
    fo, fs = feats
    full = naive_matrix(fs, fo)
    monkeypatch.setattr(config, "DISTANCE_CHUNK_BYTES", 8 * fo.n_rows * 4 * 7)  # 청크 여러 개
    idx, dist = D.nearest_neighbor(fs, fo, n_jobs=2)
    np.testing.assert_array_equal(idx, np.argmin(full, axis=1))
    np.testing.assert_allclose(dist, full.min(axis=1), atol=1e-12)


def test_nearest_neighbor_exclude_self(feats):
    fo, _ = feats
    full = naive_matrix(fo, fo)
    np.fill_diagonal(full, np.inf)
    idx, dist = D.nearest_neighbor(fo, fo, exclude=np.arange(fo.n_rows))
    np.testing.assert_array_equal(idx, np.argmin(full, axis=1))


def test_knn_lexicographic_order(feats, monkeypatch):
    fo, _ = feats
    full = naive_matrix(fo, fo)
    np.fill_diagonal(full, np.inf)
    monkeypatch.setattr(config, "DISTANCE_CHUNK_BYTES", 8 * fo.n_rows * 4 * 5)
    k = 7
    idx, dist = D.k_nearest_neighbors(fo, k, n_jobs=2)
    for r in range(fo.n_rows):
        order = np.lexsort((np.arange(fo.n_rows), full[r]))[:k]
        np.testing.assert_array_equal(idx[r], order)
        np.testing.assert_allclose(dist[r], full[r, order], atol=1e-12)


def test_cosine_rejects_missing_numeric():
    syn = make_mixed(15, 6).astype({"income": float})
    syn.loc[0, "income"] = np.nan
    data = validate_pair(make_mixed(20, 5), syn)
    with pytest.raises(ValueError, match="cosine"):
        D.prepare_features(data.original, [data.synthetic], data.schema, "cosine")


def test_use_gpu_falls_back_to_cpu(caplog):
    assert D.resolve_gpu(False, "cosine") is None
    device = D.resolve_gpu(True, "cosine")  # torch/CUDA 가 없으면 경고 후 None
    if device is None:
        assert "falling back to CPU" in caplog.text
