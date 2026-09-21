"""임계값 계산: 최적화된 시뮬레이션이 분할마다 전수 계산한 결과와 같은지, 지원 범위 검사 등."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_mixed
from test_distance import naive_matrix

from sdqe import config, io_utils
from sdqe.common.distance import prepare_features
from sdqe.metrics.privacy.single_out_risk import compute_single_out_risk
from sdqe.preprocess.validation import prepare_original
from sdqe.threshold import distribution, simulation
from sdqe.threshold.runner import ThresholdConfigError, ThresholdOptions, run_threshold


@pytest.fixture
def original():
    return prepare_original(make_mixed(48, 21))


@pytest.mark.parametrize("metric", ["gower", "cosine"])
@pytest.mark.parametrize("list_size", [64, 2])  # 2 이면 목록 밖 전수 탐색 보완 경로도 검증
def test_inference_simulation_matches_bruteforce(original, monkeypatch, metric, list_size):
    df, schema = original
    monkeypatch.setattr(config, "NEIGHBOR_LIST_SIZE", list_size)
    seeds = simulation.derive_seeds(42, 6)
    values, info = simulation.simulate_inference_risk(df, schema, seeds, metric)
    # 스케일링 기준은 원본 전체에서 한 번만 정하고 모든 분할에 재사용합니다 (명세 4.6절).
    (f,) = prepare_features(df, [], schema, metric)
    full = naive_matrix(f, f)
    expected = []
    for seed in seeds:
        a, b = simulation.split_halves(len(df), seed)
        d_ba = full[np.ix_(b, a)]
        d_aa = full[np.ix_(a, a)]
        np.fill_diagonal(d_aa, np.inf)
        pairs = [(d_ba[i].min(), d_aa[int(np.argmin(d_ba[i]))].min()) for i in range(len(b))]
        flagged = sum(d1 < d2 for d1, d2 in pairs)
        n_ties = sum(d1 == d2 for d1, d2 in pairs)
        expected.append(flagged / (len(pairs) - n_ties))
    np.testing.assert_allclose(values, expected, atol=0)
    if list_size == 2:
        assert info["n_exhaustive_fallbacks"] > 0


def test_single_out_simulation_matches_direct(original):
    df, schema = original
    seeds = simulation.derive_seeds(1, 5)
    for adj in (True, False):
        values = simulation.simulate_single_out_risk(df, schema, seeds, adj)
        for v, seed in zip(values, seeds):
            a, b = simulation.split_halves(len(df), seed)
            direct = compute_single_out_risk(df.iloc[a], df.iloc[b], schema, adj)["value"]
            assert v == pytest.approx(direct, abs=0)


def test_size_correction_formula():
    assert simulation.size_correction(0.1) == pytest.approx(1 - 0.9**2)
    np.testing.assert_allclose(simulation.size_correction(np.array([0.0, 1.0])), [0.0, 1.0])


def test_distribution_single_out_leave_one_out():
    df = pd.DataFrame({"x": ["a", "a", "a", "b", "b", "c"]})
    df, schema = prepare_original(df)
    # 나머지 원본 내 동일 레코드 수: a->2, b->1, c->0
    assert distribution.single_out_risk_proportion(df, schema, True) == pytest.approx(
        (3 * 0.5 + 2 * 1) / 6
    )
    assert distribution.single_out_risk_proportion(df, schema, False) == pytest.approx(5 / 6)


@pytest.mark.parametrize("metric", ["gower", "cosine"])
def test_distribution_inference_matches_definition(original, metric):
    df, schema = original
    p, _ = distribution.inference_risk_proportion(df, schema, metric)
    (f,) = prepare_features(df, [], schema, metric)
    d = naive_matrix(f, f)
    np.fill_diagonal(d, np.inf)
    flags, ties = [], []
    for r in range(len(df)):
        k = int(np.argmin(d[r]))
        rest = d[k].copy()
        rest[r] = np.inf
        flags.append(d[r, k] < rest.min())
        ties.append(d[r, k] == rest.min())
    assert p == pytest.approx(sum(flags) / (len(flags) - sum(ties)), abs=0)


def test_normal_quantile_threshold():
    res = distribution.normal_quantile_threshold(0.2, 1000, 0.95)
    assert res["threshold"] == pytest.approx(0.2 + 1.6448536 * np.sqrt(0.2 * 0.8 / 1000), rel=1e-6)


# ---------------------------------------------------------------- runner
def _csv(tmp_path):
    path = tmp_path / "orig.csv"
    make_mixed(40, 3).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize(
    "metric,method",
    [
        ("pmse", "distribution"),
        ("bivariate_similarity", "distribution"),
        ("linkage_risk", "simulation"),
        ("univariate_similarity", "simulation"),
    ],
)
def test_unsupported_combinations_rejected(tmp_path, metric, method):
    with pytest.raises(ThresholdConfigError):
        run_threshold(
            ThresholdOptions(
                metric=metric, original=_csv(tmp_path), method=method, output_dir=str(tmp_path)
            )
        )


def test_strict_returns_zero_without_computation(tmp_path):
    res = run_threshold(
        ThresholdOptions(
            metric="single_out_risk",
            original="does-not-exist.csv",
            strict=True,
            output_dir=str(tmp_path),
        )
    )
    assert res["threshold"] == 0.0 and res["threshold_source"] == "user_defined"


def test_simulation_outputs_all_raw_values(tmp_path, caplog):
    res = run_threshold(
        ThresholdOptions(
            metric="single_out_risk",
            original=_csv(tmp_path),
            n_simulations=7,
            output_dir=str(tmp_path),
        )
    )
    assert "below 100" in caplog.text
    raw = io_utils.load_json(res["simulation"]["simulations_file"])
    assert len(raw["values"]) == 7 and len(raw["seeds"]) == 7
    so = res["single_out_risk"]
    assert so["p_star"] == pytest.approx(1 - (1 - so["p_hat"]) ** 2)
    assert res["threshold"] == pytest.approx(so["p_star"])


def test_threshold_is_reproducible(tmp_path):
    kw = dict(
        metric="inference_risk", original=_csv(tmp_path), n_simulations=5, output_dir=str(tmp_path)
    )
    assert (
        run_threshold(ThresholdOptions(**kw))["threshold"]
        == run_threshold(ThresholdOptions(**kw))["threshold"]
    )


def test_extended_scope_note(tmp_path):
    res = run_threshold(
        ThresholdOptions(
            metric="pmse",
            original=_csv(tmp_path),
            n_simulations=3,
            output_dir=str(tmp_path),
            model="logistic_regression",
        )
    )
    assert res["note"] == config.EXTENDED_SCOPE_NOTE
