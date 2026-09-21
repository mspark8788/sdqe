"""연결위험도 ``linkage_risk`` (CAP, Correct Attribution Probability).

기존 ``cap_evaluation_.py`` 의 계산 로직을 구조만 정리해 이전했습니다. 수식은 바꾸지 않았고,
동일성은 ``tests/test_linkage_risk.py`` 에서 기존 구현과 비교해 검증합니다.

기존 구현의 계산 방식 (합성데이터 생성·활용 안내서 [1] 65~66쪽 기준)
- 원본 레코드마다 하나씩 CAP 을 계산합니다.
  ``CAP_j = (합성에서 준식별자 K 와 민감정보 T 가 모두 일치하는 레코드 수)
  / (합성에서 준식별자 K 가 일치하는 레코드 수)``
- **분모가 0 인 경우(준식별자가 일치하는 합성 레코드가 없음) CAP 은 0** 으로 두고 평균에 포함합니다.
  해당 레코드 수는 ``unmatched_count`` 로 보고합니다.
- 값은 문자열로 비교합니다. (``'01'`` 과 ``'1'``, ``'1'`` 과 ``'1.0'`` 은 서로 다른 범주)
- 빈 문자열, ``None``, ``null``, ``NULL``, ``NaN``, ``nan`` 은 결측입니다. 결측 처리
  ``missing_policy='exclude'`` 는 K 또는 평가 중인 T 가 결측인 원본 행을 평가에서 제외하고,
  합성에서도 해당 K, T 가 유효한 행만 분자와 분모에 포함합니다.
- ``evaluation_mode='combined'`` 는 민감정보별 개별 평가에 더해
  전체 민감정보 결합 CAP 도 계산합니다.
- 임계값 초과 규칙: ``cap > threshold``.

임계값은 계산하지 않으며 사용자가 직접 지정합니다(기본 0.7, 0.0 이상 1.0 미만, 명세 6.3절).

References:
    [4] Taub, J. and Elliot, M. (2019). The synthetic data challenge.
    [1] 개인정보보호위원회, 합성데이터 생성·활용 안내서.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from sdqe import config

logger = logging.getLogger("sdqe")

# 기존 구현이 결측으로 보는 문자열 표기
_MISSING_TOKENS = ["", "None", "null", "NULL", "NaN", "nan"]
# CSV 읽기 시 결측으로 보는 표기 (기존 run_cap 과 동일)
CSV_NA_VALUES = ["", "null", "NULL", "NaN", "nan"]


def validate_linkage_threshold(threshold: float) -> None:
    """``--linkage_threshold`` 범위를 확인합니다 (명세 6.3절).

    Raises:
        ValueError: 0.0 미만이거나 1.0 이상일 때.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError(
            f"--linkage_threshold must be in [0.0, 1.0); got {threshold}. "
            "A CAP threshold of 1 means sensitive values can be inferred with certainty."
        )
    if threshold < config.LINKAGE_THRESHOLD_WARN_BELOW:
        logger.warning(
            "--linkage_threshold %.3f is below %.1f: many records may exceed it and removing them "
            "can reduce utility.",
            threshold,
            config.LINKAGE_THRESHOLD_WARN_BELOW,
        )


def calculate_cap(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    key_columns: Sequence[str],
    sensitive_columns: Sequence[str],
    threshold: float | None = config.DEFAULT_LINKAGE_THRESHOLD,
    missing_policy: str = config.DEFAULT_LINKAGE_MISSING_POLICY,
    evaluation_mode: str = config.DEFAULT_LINKAGE_EVALUATION_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본 레코드별 CAP 과 민감정보별 요약을 계산합니다 (기존 ``calculate_cap`` 이전).

    Args:
        original: 원본 데이터 (문자열로 읽은 값 권장).
        synthetic: 합성 데이터 (문자열로 읽은 값 권장).
        key_columns: 준식별자 열 (K).
        sensitive_columns: 민감정보 열 (T).
        threshold: CAP 초과 판정 임계값. ``None`` 이면 판정하지 않음.
        missing_policy: ``exclude`` (결측 행 제외) 또는 ``error`` (결측 시 중단).
        evaluation_mode: ``individual`` 또는 ``combined``.

    Returns:
        (레코드별 결과, 민감정보별 요약).

    References:
        [4] Taub and Elliot (2019); [1] 합성데이터 생성·활용 안내서 65~66쪽.
    """
    if isinstance(key_columns, str) or isinstance(sensitive_columns, str):
        raise ValueError("Column arguments must be lists, not strings.")
    keys, targets = list(key_columns), list(sensitive_columns)
    columns = keys + targets
    if not keys or not targets:
        raise ValueError("Specify both --quasi_identifiers and --sensitive_attributes.")
    if len(columns) != len(set(columns)):
        raise ValueError(
            "Duplicated columns, or overlap between quasi-identifiers and sensitive attributes."
        )
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be within [0, 1] or None.")
    if missing_policy not in config.LINKAGE_MISSING_POLICIES:
        raise ValueError(f"missing_policy must be one of {config.LINKAGE_MISSING_POLICIES}.")
    if evaluation_mode not in config.LINKAGE_EVALUATION_MODES:
        raise ValueError(f"evaluation_mode must be one of {config.LINKAGE_EVALUATION_MODES}.")

    frames = []
    for label, frame in [("original", original), ("synthetic", synthetic)]:
        if frame.empty:
            raise ValueError(f"{label} data is empty.")
        if not frame.columns.is_unique:
            raise ValueError(f"{label} data has duplicated column names.")
        absent = [col for col in columns if col not in frame.columns]
        if absent:
            raise ValueError(f"Columns not found in {label} data: {absent}")
        selected = frame[columns].reset_index(drop=True).astype("string")
        selected = selected.replace(_MISSING_TOKENS, pd.NA)
        missing = selected.isna().sum()
        if missing_policy == "error" and missing.any():
            raise ValueError(
                f"Missing values in {label} data: {missing[missing > 0].to_dict()}. "
                "Preprocess them or use --linkage_missing_policy exclude."
            )
        selected = selected.fillna("None")
        frames.append(selected)
    original, synthetic = frames

    # 변수명에 한글/공백/특수문자가 있어도 안전하도록 내부 식별자로 바꿔 계산합니다.
    internal_keys = [f"_key_{i}" for i in range(len(keys))]
    evaluations = [("individual", [target]) for target in targets]
    if evaluation_mode == "combined" and len(targets) > 1:
        evaluations.append(("combined", targets))
    records, summaries = [], []
    for evaluation_type, target_columns in evaluations:
        target = " + ".join(target_columns)
        internal_targets = [f"_target_{i}" for i in range(len(target_columns))]
        rename = dict(zip(keys, internal_keys))
        rename.update(zip(target_columns, internal_targets))
        orig = original[keys + target_columns].rename(columns=rename)
        syn = synthetic[keys + target_columns].rename(columns=rename)
        orig["original_row"] = range(len(orig))
        missing_keys = orig[internal_keys].eq("None").any(axis=1)
        missing_target = orig[internal_targets].eq("None").any(axis=1)
        orig["evaluated"] = ~(missing_keys | missing_target)
        orig["exclusion_reason"] = ""
        orig.loc[missing_keys, "exclusion_reason"] = "missing_key"
        orig.loc[missing_target, "exclusion_reason"] = "missing_target"
        orig.loc[missing_keys & missing_target, "exclusion_reason"] = "missing_key_and_target"
        syn = syn.loc[~syn.eq("None").any(axis=1)]

        # 해시 기반 그룹화: 합성에서 K 일치 수, K+T 일치 수를 한 번에 센 뒤 원본에 조인
        key_counts = syn.groupby(internal_keys, dropna=False).size()
        target_counts = syn.groupby(internal_keys + internal_targets, dropna=False).size()
        detail = (
            orig.merge(
                key_counts.rename("key_match_count").reset_index(),
                on=internal_keys,
                how="left",
                validate="many_to_one",
                sort=False,
            )
            .merge(
                target_counts.rename("target_match_count").reset_index(),
                on=internal_keys + internal_targets,
                how="left",
                validate="many_to_one",
                sort=False,
            )
            .sort_values("original_row")
            .reset_index(drop=True)
        )

        for col in ["key_match_count", "target_match_count"]:
            detail[col] = detail[col].fillna(0).astype("int64")
        # 분모 0 (K 일치 합성 레코드 없음) -> CAP = 0
        detail["cap"] = (
            detail["target_match_count"] / detail["key_match_count"].replace(0, float("nan"))
        ).fillna(0.0)
        detail["matched"] = detail["key_match_count"].gt(0).astype("boolean")
        detail["sensitive_column"] = target
        detail["evaluation_type"] = evaluation_type
        detail["sensitive_columns"] = json.dumps(target_columns, ensure_ascii=False)
        detail["exceeds_threshold"] = (
            detail["cap"].gt(threshold).astype("boolean")
            if threshold is not None
            else pd.Series(pd.NA, index=detail.index, dtype="boolean")
        )
        excluded = ~detail["evaluated"]
        for col in ["key_match_count", "target_match_count"]:
            detail[col] = detail[col].astype("Int64").mask(excluded)
        detail.loc[excluded, "cap"] = float("nan")
        detail.loc[excluded, ["matched", "exceeds_threshold"]] = pd.NA
        valid = detail.loc[detail["evaluated"]]
        records.append(
            detail[
                [
                    "original_row",
                    "sensitive_column",
                    "key_match_count",
                    "target_match_count",
                    "cap",
                    "matched",
                    "exceeds_threshold",
                    "evaluated",
                    "exclusion_reason",
                    "evaluation_type",
                    "sensitive_columns",
                ]
            ]
        )
        summaries.append(
            {
                "sensitive_column": target,
                "evaluation_type": evaluation_type,
                "sensitive_columns": json.dumps(target_columns, ensure_ascii=False),
                "evaluation_mode": evaluation_mode,
                "key_columns": json.dumps(keys, ensure_ascii=False),
                "original_count": len(original),
                "synthetic_count": len(synthetic),
                "evaluated_count": len(valid),
                "excluded_count": int(excluded.sum()),
                "excluded_ratio": excluded.mean(),
                "synthetic_valid_count": len(syn),
                "synthetic_excluded_count": len(synthetic) - len(syn),
                "evaluation_status": "evaluated" if len(valid) else "no_valid_original_records",
                "mean_denominator": "evaluated_original_records",
                "mean_cap": detail["cap"].mean(),
                "max_cap": detail["cap"].max(),
                "matched_count": int(detail["matched"].sum()),
                "unmatched_count": int((~detail["matched"]).sum()),
                "unmatched_ratio": float((~valid["matched"]).mean())
                if len(valid)
                else float("nan"),
                "mean_cap_matched": detail.loc[detail["matched"], "cap"].mean(),
                "threshold": threshold,
                "threshold_rule": "cap > threshold" if threshold is not None else "disabled",
                "exceed_count": int(detail["exceeds_threshold"].sum())
                if threshold is not None
                else None,
                "exceed_ratio": float(valid["exceeds_threshold"].mean())
                if threshold is not None and len(valid)
                else None,
                "unmatched_cap": 0,
                "missing_policy": missing_policy,
                "missing_fill_value": "None",
            }
        )
    return pd.concat(records, ignore_index=True), pd.DataFrame(summaries)


def compute_linkage_risk(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str],
    linkage_threshold: float = config.DEFAULT_LINKAGE_THRESHOLD,
    missing_policy: str = config.DEFAULT_LINKAGE_MISSING_POLICY,
    evaluation_mode: str = config.DEFAULT_LINKAGE_EVALUATION_MODE,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """연결위험도를 계산하고 결과 JSON 용 요약을 만듭니다.

    ``value`` 는 평가별 평균 CAP 중 최댓값(가장 보수적인 값)이며, 평가별 값은
    ``evaluations`` 에 모두 기록합니다. 모든 평가에서 임계값을 넘는 레코드가 없으면 통과입니다.

    Args:
        original: 원본 데이터 (문자열로 읽은 값).
        synthetic: 합성 데이터 (문자열로 읽은 값).
        quasi_identifiers: 준식별자 열.
        sensitive_attributes: 민감정보 열.
        linkage_threshold: 판정 임계값 (0.0 이상 1.0 미만).
        missing_policy: 결측 처리 방침.
        evaluation_mode: 평가 방식.

    Returns:
        (요약 딕셔너리, 레코드별 결과 DataFrame).
    """
    validate_linkage_threshold(linkage_threshold)
    records, summary = calculate_cap(
        original,
        synthetic,
        quasi_identifiers,
        sensitive_attributes,
        linkage_threshold,
        missing_policy,
        evaluation_mode,
    )
    evaluations = summary.to_dict(orient="records")
    for ev in evaluations:
        ev["sensitive_columns"] = json.loads(ev["sensitive_columns"])
        ev["key_columns"] = json.loads(ev["key_columns"])
    mean_caps = [ev["mean_cap"] for ev in evaluations if pd.notna(ev["mean_cap"])]
    total_exceed = int(sum(ev["exceed_count"] or 0 for ev in evaluations))
    total_evaluated = int(sum(ev["evaluated_count"] for ev in evaluations))
    result = {
        "value": max(mean_caps) if mean_caps else float("nan"),
        "value_definition": "max of mean CAP over evaluations",
        "threshold": linkage_threshold,
        "threshold_source": config.THRESHOLD_SOURCE_USER,
        "passed": total_exceed == 0,
        # 임계값은 레코드 단위 기준입니다 ([1] 안내서 66쪽). 평균 CAP 과 비교하는 것이 아닙니다.
        "pass_rule": "no record with cap > threshold",
        "exceed_count": total_exceed,
        "exceed_ratio": total_exceed / total_evaluated if total_evaluated else None,
        "n_evaluated_records": total_evaluated,
        "denominator_zero_policy": "CAP = 0 when no synthetic record matches the quasi-identifiers "
        "(counted in the mean; reported as unmatched_count)",
        "evaluations": evaluations,
    }
    return result, records
