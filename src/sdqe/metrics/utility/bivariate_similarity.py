"""2차원 유사성 ``bivariate_similarity`` (명세 5.2절).

변수 쌍마다 연관성 계수를 구해 원본과 합성의 연관성 행렬을 만들고,
두 행렬 차이의 표준편차(모집단 표준편차, 대각 포함)를 지표값으로 씁니다.
작을수록 두 데이터의 2차원 구조가 비슷합니다. (기존 ``corMat_std`` 의 계산식 유지)

변수 쌍별 연관성 계수
- 수치형-수치형: Pearson 상관계수 (결측 쌍 제외, ``pandas.Series.corr``)
- 수치형-범주형: 일원 분산분석 OLS 의 결정계수 ``ESS / (ESS + SSR)``
- 범주형-범주형: Cramér's V (``scipy.stats.chi2_contingency`` 기본 설정)

OLS 는 문자열 formula 없이 설계행렬을 직접 구성해 ``numpy.linalg.lstsq`` 로 풉니다.
변수명에 공백, 특수문자, 한글이 있어도 동작합니다. 범주형은 정렬된 수준의 첫 수준을
기준으로 하는 더미(treatment coding)로 인코딩하며, 이는 기존 ``statsmodels`` 의
``ols('y ~ C(x)')`` 와 같은 계수와 결정계수를 줍니다.

성능: 열마다 정수 코드를 한 번만 만들고, 교차표는 ``np.bincount`` 로 셉니다
(``pd.crosstab`` 과 같은 표: 두 변수 모두 관측된 행만 쓰고 빈 행/열 없음).

기존 구현과의 차이 (사용자 확인 후 수정)
    기존 ``corMat_std`` 는 조건식 오타로 (범주형, 수치형) 순서의 쌍을 Cramér's V 로
    계산했습니다. 여기서는 그 쌍도 분산분석 결정계수로 계산합니다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from sdqe.preprocess.validation import Schema, find_constant_columns

logger = logging.getLogger("sdqe")


@dataclass
class OLSResult:
    """일원 분산분석 OLS 결과.

    Attributes:
        coefficients: 원래 이름으로 복원한 계수. ``Intercept`` 와 ``<범주형 열>[T.<수준>]``.
        r_squared: 결정계수 ``ESS / (ESS + SSR)``.
        n_obs: 사용한 관측치 수.
    """

    coefficients: dict[str, float]
    r_squared: float
    n_obs: int


def _fit_one_way(y: np.ndarray, codes: np.ndarray, n_levels: int) -> tuple[np.ndarray, float]:
    """설계행렬 [상수항, 수준 2..L 더미] 로 OLS 를 풀어 (계수, 결정계수) 를 반환합니다.

    ``codes`` 는 0..L-1 의 수준 번호이며 0 이 기준 범주입니다. 결측은 미리 제외되어 있어야 합니다.
    """
    x = np.zeros((len(y), n_levels), dtype=float)
    x[:, 0] = 1.0
    dummy_rows = np.flatnonzero(codes > 0)
    x[dummy_rows, codes[dummy_rows]] = 1.0
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    ssr = float(np.sum((y - x @ beta) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2)) if len(y) else 0.0
    ess = tss - ssr  # statsmodels 와 같은 정의 (상수항 포함 모형)
    r2 = ess / (ssr + ess) if tss > 0 else float("nan")
    return beta, r2


def ols_one_way(y: pd.Series, groups: pd.Series) -> OLSResult:
    """수치형 ``y`` 를 범주형 ``groups`` 로 회귀하는 OLS 를 설계행렬로 직접 풉니다.

    변수명은 계산에 쓰지 않고(위치 기반), 결과 계수 이름에서만 원래 이름으로 복원합니다.

    Args:
        y: 수치형 반응변수.
        groups: 범주형 설명변수.

    Returns:
        :class:`OLSResult`. 반응변수 분산이 0 이면 결정계수는 NaN.
    """
    mask = y.notna().to_numpy() & groups.notna().to_numpy()
    codes, levels = pd.factorize(groups[mask], sort=True)
    beta, r2 = _fit_one_way(y.to_numpy(dtype=float)[mask], codes, len(levels))
    names = ["Intercept"] + [f"{groups.name}[T.{lvl}]" for lvl in levels[1:]]
    return OLSResult(dict(zip(names, map(float, beta))), r2, int(mask.sum()))


def cramers_v(table: pd.DataFrame | np.ndarray) -> float:
    """교차표로 Cramér's V 를 계산합니다 (기존 ``cramerV`` 와 동일)."""
    table = np.asarray(table)
    stat = chi2_contingency(table)[0]
    n = np.sum(table)
    min_dim = min(table.shape) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sqrt((np.float64(stat) / n) / np.float64(min_dim)))


@dataclass
class _EncodedColumn:
    is_categorical: bool
    values: np.ndarray  # 수치형: float, 범주형: 정렬된 수준 번호 (결측 -1)
    n_levels: int
    series: pd.Series


def _encode(df: pd.DataFrame, columns: Sequence[str], schema: Schema) -> list[_EncodedColumn]:
    encoded = []
    for col in columns:
        s = df[col]
        if schema.is_categorical(col):
            codes, levels = pd.factorize(s, sort=True)
            encoded.append(_EncodedColumn(True, codes, len(levels), s))
        else:
            encoded.append(_EncodedColumn(False, s.to_numpy(dtype=float), 0, s))
    return encoded


def _anova_r2(num: _EncodedColumn, cat: _EncodedColumn) -> float:
    mask = ~np.isnan(num.values) & (cat.values >= 0)
    present, codes = np.unique(cat.values[mask], return_inverse=True)  # 관측된 수준만, 정렬 유지
    return _fit_one_way(num.values[mask], codes, len(present))[1]


def _cramers_v_codes(a: _EncodedColumn, b: _EncodedColumn) -> float:
    mask = (a.values >= 0) & (b.values >= 0)
    table = np.bincount(
        a.values[mask] * b.n_levels + b.values[mask], minlength=a.n_levels * b.n_levels
    ).reshape(a.n_levels, b.n_levels)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    return cramers_v(table)


def _pair(a: _EncodedColumn, b: _EncodedColumn) -> tuple[float, str]:
    if not a.is_categorical and not b.is_categorical:
        return float(a.series.corr(b.series)), "pearson"
    if not a.is_categorical:
        return _anova_r2(a, b), "anova_r_squared"
    if not b.is_categorical:
        return _anova_r2(b, a), "anova_r_squared"
    return _cramers_v_codes(a, b), "cramers_v"


def pair_association(a: pd.Series, b: pd.Series, a_cat: bool, b_cat: bool) -> tuple[float, str]:
    """변수 쌍의 연관성 계수와 계산 방식을 반환합니다."""
    frame = pd.DataFrame({"a": a.to_numpy(), "b": b.to_numpy()})
    schema = Schema(
        ("a", "b"),
        tuple(n for n, c in (("a", a_cat), ("b", b_cat)) if c),
        tuple(n for n, c in (("a", a_cat), ("b", b_cat)) if not c),
    )
    ea, eb = _encode(frame, ["a", "b"], schema)
    return _pair(ea, eb)


def association_matrix(
    df: pd.DataFrame, columns: Sequence[str], schema: Schema
) -> tuple[np.ndarray, list[str]]:
    """연관성 행렬(대각 1)과 쌍별 계산 방식 목록을 만듭니다."""
    encoded = _encode(df, columns, schema)
    p = len(encoded)
    mat = np.eye(p)
    kinds = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(p - 1):
            for j in range(i + 1, p):
                value, kind = _pair(encoded[i], encoded[j])
                mat[i, j] = mat[j, i] = value
                kinds.append(kind)
    return mat, kinds


def constant_columns_of(
    original: pd.DataFrame, synthetic: pd.DataFrame, schema: Schema
) -> list[str]:
    """원본 또는 합성에서 상수인 열 (스키마 순서)."""
    constant = set(find_constant_columns(original, schema.columns)) | set(
        find_constant_columns(synthetic, schema.columns)
    )
    return [c for c in schema.columns if c in constant]


def compute_bivariate_similarity(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
    drop_constant_columns: bool = True,
    include_pairs: bool = True,
) -> dict[str, Any]:
    """2차원 유사성을 계산합니다.

    Args:
        original: 원본 데이터 (타입 정리 완료).
        synthetic: 합성 데이터 (타입 정리 완료).
        schema: 분석 스키마.
        drop_constant_columns: 원본 또는 합성에서 상수인 열을 제외할지 여부.
        include_pairs: 결과에 쌍별 계수를 포함할지 여부.

    Returns:
        ``value`` (차이 행렬 표준편차), ``excluded_columns``, ``pairs`` 등을 담은 딕셔너리.
        계산이 불가능한 쌍이 있으면 ``value`` 가 NaN 이 됩니다.
    """
    constant = constant_columns_of(original, synthetic, schema)
    columns = list(schema.columns)
    excluded: list[str] = []
    if constant and drop_constant_columns:
        excluded = constant
        columns = [c for c in columns if c not in constant]
        logger.info("bivariate_similarity: excluded constant columns %s", constant)
    elif constant:
        logger.warning(
            "bivariate_similarity: constant columns %s may produce NaN; "
            "use --drop_constant_columns on.",
            constant,
        )
    if len(columns) < 2:
        raise ValueError("bivariate_similarity needs at least two non-constant columns.")
    mat_o, kinds = association_matrix(original, columns, schema)
    mat_s, _ = association_matrix(synthetic, columns, schema)
    value = float(np.std(mat_o - mat_s))
    result: dict[str, Any] = {
        "value": value,
        "excluded_columns": excluded,
        "n_columns_used": len(columns),
        "n_nan_pairs": int(np.isnan(np.triu(mat_o, 1)).sum() + np.isnan(np.triu(mat_s, 1)).sum()),
    }
    if include_pairs:
        pairs = []
        k = 0
        for i in range(len(columns) - 1):
            for j in range(i + 1, len(columns)):
                pairs.append(
                    {
                        "column_1": columns[i],
                        "column_2": columns[j],
                        "method": kinds[k],
                        "original": mat_o[i, j],
                        "synthetic": mat_s[i, j],
                        "difference": mat_o[i, j] - mat_s[i, j],
                    }
                )
                k += 1
        result["pairs"] = pairs
    return result
