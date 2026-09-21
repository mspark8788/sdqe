"""전처리: 정합성 검증과 초과 레코드 삭제 전략."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_mixed

from sdqe.preprocess.reduction import (
    REDUCTION_STRATEGIES,
    ReductionContext,
    plan_inference_deletions,
    reduce_records,
)
from sdqe.preprocess.validation import DataValidationError, validate_pair


# ---------------------------------------------------------------- validation
def test_name_mismatch_is_reported():
    o, s = make_mixed(10, 1), make_mixed(10, 2).rename(columns={"age": "AGE"})
    with pytest.raises(DataValidationError) as exc:
        validate_pair(o, s)
    assert "Missing in synthetic: ['age']" in str(exc.value)
    assert "Missing in original: ['AGE']" in str(exc.value)


def test_order_is_aligned_and_recorded():
    o, s = make_mixed(10, 1), make_mixed(10, 2)
    res = validate_pair(o, s[list(reversed(s.columns))])
    assert res.report["column_order_reordered"] is True
    assert list(res.synthetic.columns) == list(o.columns)


def test_type_mismatch_is_error_unless_declared_categorical():
    o, s = make_mixed(10, 1), make_mixed(10, 2)
    s["age"] = s["age"].astype(str) + "세"
    with pytest.raises(DataValidationError, match="'age'"):
        validate_pair(o, s)
    res = validate_pair(o, s, categorical_columns=["age"])
    assert "age" in res.schema.categorical


def test_integer_codes_match_float_codes_as_categories():
    o = pd.DataFrame({"code": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    s = pd.DataFrame({"code": [1.0, 2.0, np.nan], "x": [1.0, 2.0, 3.0]})
    res = validate_pair(o, s, categorical_columns=["code"])
    assert res.synthetic["code"].tolist()[:2] == ["1", "2"]


def test_format_checks():
    o, s = make_mixed(20, 1), make_mixed(20, 2)
    s.loc[0, "sex"] = "?"
    s.loc[1, "region"] = "제주"
    rep = validate_pair(o, s).report["format"]
    assert rep["missing_like_tokens"]["sex"]["synthetic"] == {"?": 1}
    assert "제주" in rep["unseen_categories"]["region"]["values"]


def test_exclude_columns():
    res = validate_pair(make_mixed(10, 1), make_mixed(10, 2), exclude_columns=["grade"])
    assert "grade" not in res.schema.columns and res.report["excluded_columns"] == ["grade"]


# ---------------------------------------------------------------- reduction
def test_registry_has_three_strategies():
    assert set(REDUCTION_STRATEGIES) == {"random", "pmse_probability", "inference_distance"}


def test_inference_plan_hits_target_when_feasible():
    """목표 위험도가 달성 가능하면 정확히 목표 비율이 되도록 삭제한다."""
    rng = np.random.default_rng(3)
    n, target_size = 300, 200
    flags = rng.random(n) < 0.6
    ds = np.round(rng.random(n), 3)
    delete, info = plan_inference_deletions(flags, ds, 0.45, target_size)
    assert info["target_feasible"]
    assert len(delete) == n - target_size
    assert len(set(delete.tolist())) == len(delete)
    assert info["inference_risk_after"] == pytest.approx(round(0.45 * target_size) / target_size)
    kept = np.delete(flags, delete)
    assert kept.sum() / len(kept) == pytest.approx(info["inference_risk_after"])


def test_inference_plan_moves_toward_target_when_infeasible():
    flags = np.array([1] * 216 + [0] * 75, dtype=bool)
    ds = np.linspace(0, 1, 291)
    delete, info = plan_inference_deletions(flags, ds, 0.4655, 200)
    assert not info["target_feasible"]
    assert info["flag0_deleted"] == 0 and len(delete) == 91
    assert info["inference_risk_after"] < info["inference_risk_before"]


def _ctx(seed=0):
    data = validate_pair(make_mixed(40, 1), make_mixed(55, 2))
    return ReductionContext(
        original=data.original,
        synthetic=data.synthetic,
        synthetic_full=data.synthetic,
        rows=np.arange(len(data.synthetic)),
        schema=data.schema,
        target_size=40,
        random_seed=seed,
        options={
            "inference_threshold": 0.5,
            "distance_metric": "gower",
            "model": "logistic_regression",
        },
    )


@pytest.mark.parametrize("method", ["random", "pmse_probability", "inference_distance"])
def test_strategies_reach_target_size_and_are_reproducible(method):
    a = reduce_records(method, _ctx())
    b = reduce_records(method, _ctx())
    assert len(a.delete_positions) == 15
    np.testing.assert_array_equal(a.delete_positions, b.delete_positions)


def test_pmse_probability_deletes_most_distinguishable():
    from sdqe.metrics.utility.pmse import compute_propensity_scores

    ctx = _ctx()
    res = reduce_records("pmse_probability", ctx)
    prob = compute_propensity_scores(
        ctx.original, ctx.synthetic, ctx.schema, "logistic_regression", None, 0
    ).synthetic_probability
    kept = np.delete(prob, res.delete_positions)
    assert prob[res.delete_positions].min() >= kept.max()
