"""구별위험도 ``single_out_risk`` (참고문헌 [2] 식 (1), (2)).

재현 데이터의 레코드와 동일한 원본 레코드를 찾을 수 있는 비율입니다.

    식 (1) 단순형:      p_hat = (1/n) * sum_i I(s_i in R)
    식 (2) 중복 보정형: p_hat = (1/n) * sum_i I(s_i in R) / d_i

``d_i`` 는 원본 데이터 안에서 재현 레코드 ``s_i`` 와 동일한 레코드의 개수입니다.
유일한 레코드는 노출 위험이 크고, 중복이 많으면 위험이 낮아지는 성질을 반영합니다.
기본은 식 (2) (``duplicate_adjustment=True``) 입니다.

행 단위 비교 대신 전체 행을 그룹 번호로 바꾼 뒤(해시 기반 그룹화) ``bincount`` 로
원본 내 중복 수 ``d_i`` 를 같은 연산에서 함께 얻습니다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from sdqe.preprocess.validation import Schema, align_column_order, check_column_names

logger = logging.getLogger("sdqe")


def record_group_ids(
    frames: Sequence[pd.DataFrame], columns: Sequence[str]
) -> tuple[list[np.ndarray], int]:
    """여러 데이터의 레코드를 공통 그룹 번호로 바꿉니다. 모든 값이 같은 레코드는 같은 번호입니다.

    결측값끼리는 같은 값으로 봅니다.

    Args:
        frames: 같은 열 구성을 가진 데이터들.
        columns: 비교할 열.

    Returns:
        (데이터별 그룹 번호 배열 목록, 전체 그룹 수).
    """
    columns = list(columns)
    combined = pd.concat([f[columns] for f in frames], ignore_index=True)
    gid = combined.groupby(columns, dropna=False, sort=False).ngroup().to_numpy(dtype=np.int64)
    out = []
    start = 0
    for f in frames:
        out.append(gid[start : start + len(f)])
        start += len(f)
    return out, int(gid.max()) + 1 if len(gid) else 0


def single_out_terms(duplicate_counts: np.ndarray, duplicate_adjustment: bool) -> np.ndarray:
    """레코드별 식 (1)/(2) 항을 계산합니다.

    Args:
        duplicate_counts: 레코드별 동일 원본 레코드 수 ``d_i`` (0 이면 일치 없음).
        duplicate_adjustment: True 면 식 (2) ``I / d_i``, False 면 식 (1) ``I``.

    Returns:
        레코드별 항 (평균이 p_hat).
    """
    d = np.asarray(duplicate_counts, dtype=np.float64)
    if duplicate_adjustment:
        return np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    return (d > 0).astype(np.float64)


def compute_single_out_risk(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
    duplicate_adjustment: bool = True,
) -> dict[str, Any]:
    """구별위험도를 계산합니다.

    계산 전에 변수 이름과 순서가 통일되어 있는지만 확인합니다(전처리 정합성 검증 유틸 재사용).

    Args:
        original: 원본 데이터 (타입 정리 완료).
        synthetic: 합성 데이터 (타입 정리 완료).
        schema: 분석 스키마.
        duplicate_adjustment: True 면 식 (2), False 면 식 (1).

    Returns:
        ``value`` (p_hat), ``n_matched``, ``formula`` 등을 담은 딕셔너리.

    References:
        [2] 박민수 외 (2024), 2.1절 식 (1), (2).
    """
    check_column_names(original, synthetic)
    synthetic, _ = align_column_order(original, synthetic)
    (gid_orig, gid_syn), n_groups = record_group_ids([original, synthetic], schema.columns)
    counts = np.bincount(gid_orig, minlength=n_groups)
    d = counts[gid_syn]
    terms = single_out_terms(d, duplicate_adjustment)
    value = float(terms.mean()) if len(terms) else float("nan")
    n_matched = int((d > 0).sum())
    return {
        "value": value,
        "formula": "equation_2" if duplicate_adjustment else "equation_1",
        "duplicate_adjustment": duplicate_adjustment,
        "n_matched_records": n_matched,
        "matched_ratio": n_matched / len(d) if len(d) else float("nan"),
        "n_original_duplicate_records": int((counts[gid_orig] > 1).sum()),
    }


def leave_one_out_duplicate_counts(original: pd.DataFrame, schema: Schema) -> np.ndarray:
    """각 원본 레코드를 재현 레코드로 보고, 나머지 원본에서 동일 레코드 수를 셉니다.

    분포가정 방식 임계값(명세 4.3절 단계 1)에서 씁니다.

    Returns:
        레코드별 나머지 원본 내 동일 레코드 수 (자기 자신 제외).
    """
    (gid,), n_groups = record_group_ids([original], schema.columns)
    counts = np.bincount(gid, minlength=n_groups)
    return counts[gid] - 1
