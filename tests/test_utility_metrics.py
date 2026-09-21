"""유용성 지표: OLS/Cramer's V/pMSE 계산, 한글·특수문자 변수명, 상수열 처리."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_mixed

statsmodels = pytest.importorskip("statsmodels")
from statsmodels.formula.api import ols  # noqa: E402

from sdqe.metrics.utility.bivariate_similarity import (  # noqa: E402
    compute_bivariate_similarity,
    cramers_v,
    ols_one_way,
    pair_association,
)
from sdqe.metrics.utility.pmse import compute_pmse, compute_propensity_scores  # noqa: E402
from sdqe.metrics.utility.univariate_similarity import compute_univariate_similarity  # noqa: E402
from sdqe.preprocess.validation import validate_pair  # noqa: E402


# ---------------------------------------------------------------- OLS
def test_ols_matches_statsmodels_coefficients_and_r2():
    df = make_mixed(80, 3)
    model = ols("income ~ C(region)", df).fit()
    ours = ols_one_way(df["income"], df["region"])
    np.testing.assert_allclose(
        list(ours.coefficients.values()), model.params.to_numpy(), rtol=1e-10
    )
    assert ours.r_squared == pytest.approx(model.ess / (model.ssr + model.ess), rel=1e-10)
    assert list(ours.coefficients)[1].startswith("region[T.")


def test_ols_works_with_korean_and_special_column_names():
    df = make_mixed(80, 3, with_names=True)
    ref = ols("y ~ C(g)", pd.DataFrame({"y": df["소득/월"], "g": df["지역-코드"]})).fit()
    ours = ols_one_way(df["소득/월"], df["지역-코드"])
    assert ours.r_squared == pytest.approx(ref.rsquared, rel=1e-10)
    assert "지역-코드[T." in list(ours.coefficients)[1]


def test_cramers_v_bincount_matches_crosstab():
    df = make_mixed(90, 4)
    value, kind = pair_association(df["sex"], df["grade"], True, True)
    assert kind == "cramers_v"
    assert value == pytest.approx(cramers_v(pd.crosstab(df["sex"], df["grade"])), abs=1e-15)


# ---------------------------------------------------------------- corMat_std
def test_bivariate_uses_anova_for_categorical_numeric_pairs():
    """범주형-수치형 쌍은 열 순서와 무관하게 일원 분산분석 결정계수를 쓴다."""
    df_o, df_s = make_mixed(70, 5), make_mixed(70, 6)
    cols = ["sex", "age", "region", "income", "grade"]  # 범주형이 수치형보다 앞
    data = validate_pair(df_o[cols], df_s[cols])
    ours = compute_bivariate_similarity(data.original, data.synthetic, data.schema)
    methods = {(p["column_1"], p["column_2"]): p["method"] for p in ours["pairs"]}
    assert methods[("sex", "age")] == "anova_r_squared"

    reordered = list(data.schema.numeric) + list(data.schema.categorical)
    other = validate_pair(df_o[reordered], df_s[reordered])
    swapped = compute_bivariate_similarity(other.original, other.synthetic, other.schema)
    assert ours["value"] == pytest.approx(swapped["value"], rel=1e-12)


def test_bivariate_drops_constant_columns():
    o, s = make_mixed(50, 1), make_mixed(50, 2)
    o["상수 열"] = 1
    s["상수 열"] = 1
    data = validate_pair(o, s)
    res = compute_bivariate_similarity(
        data.original, data.synthetic, data.schema, drop_constant_columns=True
    )
    assert res["excluded_columns"] == ["상수 열"]
    assert np.isfinite(res["value"])
    res_keep = compute_bivariate_similarity(
        data.original, data.synthetic, data.schema, drop_constant_columns=False
    )
    assert np.isnan(res_keep["value"])


# ---------------------------------------------------------------- pMSE
@pytest.mark.parametrize("model", ["decision_tree", "logistic_regression", "gradient_boosting"])
def test_pmse_models_are_reproducible(mixed_pair, model):
    a = compute_pmse(mixed_pair.original, mixed_pair.synthetic, mixed_pair.schema, model, {}, 7)
    b = compute_pmse(mixed_pair.original, mixed_pair.synthetic, mixed_pair.schema, model, {}, 7)
    assert a == b and 0 <= a["value"] <= 0.25 + 1e-12


def test_pmse_model_params_and_errors(mixed_pair):
    scores = compute_propensity_scores(
        mixed_pair.original,
        mixed_pair.synthetic,
        mixed_pair.schema,
        "decision_tree",
        {"max_depth": 2},
        1,
    )
    assert scores.synthetic_probability.shape == (len(mixed_pair.synthetic),)
    with pytest.raises(ValueError, match="Invalid --model_params"):
        compute_pmse(
            mixed_pair.original,
            mixed_pair.synthetic,
            mixed_pair.schema,
            "decision_tree",
            {"no_such_param": 1},
        )


# ---------------------------------------------------------------- univariate
def test_univariate_tests_and_figures(tmp_path):
    data = validate_pair(make_mixed(60, 1, with_names=True), make_mixed(50, 2, with_names=True))
    res = compute_univariate_similarity(data.original, data.synthetic, data.schema, tmp_path)
    assert res["variables"]["나이 (세)"]["test"] == "ks"
    assert res["variables"]["성별"]["test"] == "chi_square"
    assert (tmp_path / "univariate_similarity_summary.png").is_file()
    assert (tmp_path / "univariate_similarity_소득_월.png").is_file()
    assert len(res["figures"]) == len(data.schema.columns) + 1
