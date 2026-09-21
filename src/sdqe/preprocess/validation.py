"""원본/합성 데이터 정합성 검증 (명세 3.1절).

변수 이름, 순서, 타입, 데이터 형식 네 가지를 확인합니다.

- 이름 불일치, 타입(범주형/수치형) 불일치: 자동 보정하지 않고 :class:`DataValidationError`.
- 순서 불일치: 원본 순서로 자동 정렬하고 로그와 결과에 기록.
- 형식(결측 표기, 범주 수준 집합, 수치 범위): 결과에 경고로 기록.

여기서 정의한 :class:`Schema` 는 모든 지표 모듈이 공유합니다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sdqe import config

logger = logging.getLogger("sdqe")

_MAX_REPORTED_VALUES = 10


class DataValidationError(ValueError):
    """원본/합성 데이터 정합성 검증 실패."""


@dataclass(frozen=True)
class Schema:
    """분석 대상 열과 범주형/수치형 구분.

    Attributes:
        columns: 원본 기준 열 순서.
        categorical: 범주형 열 (``columns`` 순서 유지).
        numeric: 수치형 열 (``columns`` 순서 유지).
    """

    columns: tuple[str, ...]
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]

    def is_categorical(self, column: str) -> bool:
        """열이 범주형이면 True."""
        return column in self.categorical

    def subset(self, columns: Sequence[str]) -> Schema:
        """주어진 열만 남긴 스키마를 반환합니다 (원래 순서 유지)."""
        keep = set(columns)
        return Schema(
            columns=tuple(c for c in self.columns if c in keep),
            categorical=tuple(c for c in self.categorical if c in keep),
            numeric=tuple(c for c in self.numeric if c in keep),
        )

    def to_dict(self) -> dict[str, list[str]]:
        """JSON 기록용 딕셔너리."""
        return {
            "columns": list(self.columns),
            "categorical": list(self.categorical),
            "numeric": list(self.numeric),
        }


@dataclass
class ValidationResult:
    """정합성 검증 결과.

    Attributes:
        original: 타입이 정리된 원본 (분석 대상 열만).
        synthetic: 타입이 정리되고 원본 순서로 정렬된 합성 (분석 대상 열만).
        synthetic_raw: 원본 열 순서로 정렬만 한 합성 원자료 (제외 열 포함, 값 변환 없음).
        schema: 분석 대상 열 스키마.
        report: ``validation.json`` 으로 저장할 검증 결과.
    """

    original: pd.DataFrame
    synthetic: pd.DataFrame
    synthetic_raw: pd.DataFrame
    schema: Schema
    report: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 타입 판별 / 변환
# ---------------------------------------------------------------------------
def is_numeric_series(series: pd.Series) -> bool:
    """bool 을 제외한 수치형 dtype 이면 True. pandas 버전과 무관하게 동작합니다."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _dtype_kind(series: pd.Series) -> str:
    return "numeric" if is_numeric_series(series) else "categorical"


def to_category_strings(series: pd.Series) -> pd.Series:
    """범주형 열을 문자열(object dtype)로 통일합니다. 결측은 NaN 으로 유지합니다.

    정수값만 있는 실수 열(예: ``3.0``)은 ``"3"`` 으로 바꿔, 결측 때문에 float 로 읽힌
    정수 코드가 다른 쪽의 정수 코드와 같은 범주가 되게 합니다.

    Args:
        series: 변환할 열.

    Returns:
        문자열 값과 NaN 으로 이루어진 object dtype Series.
    """
    values = series.to_numpy(dtype=object)
    mask = pd.isna(series).to_numpy()
    out = np.empty(len(values), dtype=object)
    integral = False
    if is_numeric_series(series):
        non_null = series[~mask].astype(float)
        integral = bool(len(non_null)) and bool(np.all(np.mod(non_null.to_numpy(), 1) == 0))
    for i, (value, missing) in enumerate(zip(values, mask)):
        if missing:
            out[i] = np.nan
        elif integral:
            out[i] = str(int(value))
        elif isinstance(value, str):
            out[i] = value
        else:
            out[i] = str(value)
    return pd.Series(out, index=series.index, name=series.name, dtype=object)


def _to_numeric(series: pd.Series, label: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    bad = series[converted.isna() & series.notna()]
    if len(bad):
        samples = list(pd.unique(bad.astype(str)))[:_MAX_REPORTED_VALUES]
        raise DataValidationError(
            f"Column '{series.name}' in {label} data is expected to be numeric but contains "
            f"non-numeric values: {samples}. Declare it with --categorical_columns or clean it."
        )
    return converted.astype(float) if converted.dtype == object else converted


def coerce_types(df: pd.DataFrame, schema: Schema, label: str) -> pd.DataFrame:
    """스키마에 맞게 열 타입을 통일합니다.

    Args:
        df: 대상 데이터.
        schema: 분석 스키마.
        label: 에러 메시지에 쓸 데이터 이름(``original``/``synthetic``).

    Returns:
        범주형은 문자열(object), 수치형은 숫자 dtype 으로 바꾼 새 DataFrame.
    """
    out = pd.DataFrame(index=df.index)
    for col in schema.columns:
        if schema.is_categorical(col):
            out[col] = to_category_strings(df[col])
        else:
            out[col] = _to_numeric(df[col], label)
    return out


# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------
def apply_exclusions(
    df: pd.DataFrame, exclude_columns: Sequence[str] | None, label: str
) -> pd.DataFrame:
    """제외 열을 뺀 DataFrame 을 반환합니다. 데이터에 없는 제외 열은 경고만 남깁니다."""
    if not exclude_columns:
        return df
    missing = [c for c in exclude_columns if c not in df.columns]
    if missing:
        logger.warning("Excluded columns not found in %s data: %s", label, missing)
    return df.drop(columns=[c for c in exclude_columns if c in df.columns])


def infer_schema(
    original: pd.DataFrame, categorical_columns: Sequence[str] | None = None
) -> Schema:
    """원본 데이터로 분석 스키마를 만듭니다.

    ``categorical_columns`` 를 주면 그 열과 비수치형 dtype 열을 범주형으로,
    주지 않으면 dtype 으로 자동 추론합니다(수치형 dtype 이 아니면 범주형).

    Args:
        original: 원본 데이터 (제외 열을 뺀 상태).
        categorical_columns: 범주형으로 강제할 열 목록.

    Returns:
        분석 스키마.

    Raises:
        DataValidationError: 지정한 범주형 열이 데이터에 없을 때.
    """
    columns = [str(c) for c in original.columns]
    if len(set(columns)) != len(columns):
        dup = sorted({c for c in columns if columns.count(c) > 1})
        raise DataValidationError(f"Duplicated column names in original data: {dup}")
    explicit = list(categorical_columns or [])
    unknown = [c for c in explicit if c not in columns]
    if unknown:
        raise DataValidationError(f"--categorical_columns not found in original data: {unknown}")
    categorical = []
    auto_added = []
    for col in columns:
        if col in explicit:
            categorical.append(col)
        elif not is_numeric_series(original[col]):
            categorical.append(col)
            if explicit:
                auto_added.append(col)
    if auto_added:
        logger.info("Non-numeric columns treated as categorical as well: %s", auto_added)
    numeric = [c for c in columns if c not in categorical]
    return Schema(tuple(columns), tuple(categorical), tuple(numeric))


def prepare_original(
    original: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, Schema]:
    """원본만 쓰는 단계(임계값 계산)를 위해 원본을 정리합니다.

    Args:
        original: 원본 데이터.
        categorical_columns: 범주형으로 강제할 열.
        exclude_columns: 제외할 열.

    Returns:
        (타입이 정리된 원본, 스키마).
    """
    original = apply_exclusions(original, exclude_columns, "original")
    schema = infer_schema(original, categorical_columns)
    return coerce_types(original, schema, "original").reset_index(drop=True), schema


# ---------------------------------------------------------------------------
# 개별 검증 단계
# ---------------------------------------------------------------------------
def check_column_names(original: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    """두 데이터의 변수 이름 집합이 같은지 확인합니다.

    Raises:
        DataValidationError: 한쪽에만 있는 변수가 있을 때. 어느 쪽에 무엇이 없는지 적습니다.
    """
    orig_cols = [str(c) for c in original.columns]
    syn_cols = [str(c) for c in synthetic.columns]
    for label, cols in (("original", orig_cols), ("synthetic", syn_cols)):
        if len(set(cols)) != len(cols):
            dup = sorted({c for c in cols if cols.count(c) > 1})
            raise DataValidationError(f"Duplicated column names in {label} data: {dup}")
    only_orig = [c for c in orig_cols if c not in set(syn_cols)]
    only_syn = [c for c in syn_cols if c not in set(orig_cols)]
    if only_orig or only_syn:
        lines = ["Column names differ between original and synthetic data."]
        if only_orig:
            lines.append(f"  Missing in synthetic: {only_orig}")
        if only_syn:
            lines.append(f"  Missing in original: {only_syn}")
        raise DataValidationError("\n".join(lines))


def align_column_order(
    original: pd.DataFrame, synthetic: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    """합성 데이터의 열 순서를 원본 순서로 맞춥니다.

    이름 집합이 같다는 전제에서 호출합니다(:func:`check_column_names`).

    Returns:
        (정렬된 합성 데이터, 정렬 여부).
    """
    orig_cols = list(original.columns)
    if list(synthetic.columns) == orig_cols:
        return synthetic, False
    logger.info("Synthetic column order differs from original; reordered to original order.")
    return synthetic[orig_cols], True


def check_column_types(
    original_raw: pd.DataFrame,
    synthetic_raw: pd.DataFrame,
    schema: Schema,
    explicit_categorical: Sequence[str] | None,
) -> list[dict[str, str]]:
    """열별 타입(범주형/수치형)이 일치하는지 확인합니다.

    ``--categorical_columns`` 로 명시한 열은 양쪽을 문자열로 통일하므로 dtype 차이를 허용합니다.
    수치형끼리의 정수/실수 차이는 경고로만 남깁니다.

    Returns:
        경고 목록.

    Raises:
        DataValidationError: 범주형/수치형 구분이 다른 열이 있을 때.
    """
    explicit = set(explicit_categorical or [])
    mismatches = []
    warnings = []
    for col in schema.columns:
        o, s = original_raw[col], synthetic_raw[col]
        o_kind, s_kind = _dtype_kind(o), _dtype_kind(s)
        if o_kind != s_kind and col not in explicit:
            mismatches.append(
                f"  '{col}': original={o.dtype} ({o_kind}), synthetic={s.dtype} ({s_kind})"
            )
        elif o_kind == s_kind == "numeric" and str(o.dtype) != str(s.dtype):
            warnings.append(
                {
                    "column": col,
                    "type": "numeric_dtype_differs",
                    "detail": f"original={o.dtype}, synthetic={s.dtype}",
                }
            )
    if mismatches:
        raise DataValidationError(
            "Column types differ between original and synthetic data "
            "(categorical vs numeric):\n"
            + "\n".join(mismatches)
            + "\nFix the data, or list the columns in --categorical_columns to compare them as "
            "categories."
        )
    return warnings


def check_data_format(
    original_raw: pd.DataFrame,
    synthetic_raw: pd.DataFrame,
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
) -> dict[str, Any]:
    """결측 표기, 범주 수준 집합, 수치 범위를 비교합니다.

    Returns:
        ``missing_counts``, ``missing_like_tokens``, ``unseen_categories``,
        ``unused_categories``, ``numeric_out_of_range`` 를 담은 딕셔너리.
    """
    missing_counts = {
        col: {
            "original": int(original[col].isna().sum()),
            "synthetic": int(synthetic[col].isna().sum()),
        }
        for col in schema.columns
        if original[col].isna().any() or synthetic[col].isna().any()
    }
    tokens = set(config.MISSING_LIKE_TOKENS)
    missing_like = {}
    unseen = {}
    unused = {}
    for col in schema.categorical:
        for label, raw in (("original", original_raw[col]), ("synthetic", synthetic_raw[col])):
            as_str = raw[raw.notna()].astype(str)
            hits = as_str[as_str.isin(tokens) | (as_str.str.strip() == "")]
            if len(hits):
                missing_like.setdefault(col, {})[label] = {
                    str(k): int(v) for k, v in hits.value_counts().items()
                }
        o_levels = set(original[col].dropna())
        s_levels = set(synthetic[col].dropna())
        only_syn = sorted(s_levels - o_levels)
        only_orig = sorted(o_levels - s_levels)
        if only_syn:
            unseen[col] = {"count": len(only_syn), "values": only_syn[:_MAX_REPORTED_VALUES]}
        if only_orig:
            unused[col] = {"count": len(only_orig), "values": only_orig[:_MAX_REPORTED_VALUES]}
    out_of_range = {}
    for col in schema.numeric:
        lo, hi = original[col].min(), original[col].max()
        vals = synthetic[col].dropna()
        n_out = int(((vals < lo) | (vals > hi)).sum())
        if n_out:
            out_of_range[col] = {"count": n_out, "original_min": lo, "original_max": hi}
    return {
        "missing_counts": missing_counts,
        "missing_like_tokens": missing_like,
        "unseen_categories": unseen,
        "unused_categories": unused,
        "numeric_out_of_range": out_of_range,
    }


def find_constant_columns(df: pd.DataFrame, columns: Sequence[str] | None = None) -> list[str]:
    """모든 값이 같은(결측 포함) 열 목록을 반환합니다."""
    cols = list(columns) if columns is not None else list(df.columns)
    return [c for c in cols if df[c].nunique(dropna=False) <= 1]


# ---------------------------------------------------------------------------
# 통합 검증
# ---------------------------------------------------------------------------
def validate_pair(
    original_raw: pd.DataFrame,
    synthetic_raw: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
) -> ValidationResult:
    """원본/합성 데이터 정합성을 검증하고 분석용 데이터를 만듭니다.

    Args:
        original_raw: 원본 데이터.
        synthetic_raw: 합성 데이터.
        categorical_columns: 범주형으로 강제할 열.
        exclude_columns: 계산에서 제외할 열.

    Returns:
        :class:`ValidationResult`.

    Raises:
        DataValidationError: 이름 또는 타입이 일치하지 않을 때.
    """
    check_column_names(original_raw, synthetic_raw)
    synthetic_aligned, reordered = align_column_order(original_raw, synthetic_raw)

    original_used = apply_exclusions(original_raw, exclude_columns, "original")
    synthetic_used = apply_exclusions(synthetic_aligned, exclude_columns, "synthetic")
    schema = infer_schema(original_used, categorical_columns)
    type_warnings = check_column_types(original_used, synthetic_used, schema, categorical_columns)

    original = coerce_types(original_used, schema, "original").reset_index(drop=True)
    synthetic = coerce_types(synthetic_used, schema, "synthetic").reset_index(drop=True)
    data_format = check_data_format(original_used, synthetic_used, original, synthetic, schema)

    constant = {
        "original": find_constant_columns(original, schema.columns),
        "synthetic": find_constant_columns(synthetic, schema.columns),
    }
    report = {
        "passed": True,
        "column_names_match": True,
        "column_order_reordered": reordered,
        "schema": schema.to_dict(),
        "excluded_columns": [c for c in (exclude_columns or []) if c in original_raw.columns],
        "type_warnings": type_warnings,
        "format": data_format,
        "constant_columns": constant,
        "input_info": {
            "n_original": len(original),
            "n_synthetic": len(synthetic),
            "n_columns": len(schema.columns),
        },
    }
    for key in ("missing_like_tokens", "unseen_categories"):
        if data_format[key]:
            logger.warning("Format check: %s found in columns %s", key, list(data_format[key]))
    return ValidationResult(
        original=original,
        synthetic=synthetic,
        synthetic_raw=synthetic_aligned.reset_index(drop=True),
        schema=schema,
        report=report,
    )
