"""초과 레코드 삭제 전략 (명세 3.2절).

합성 데이터가 원본보다 클 때 원본 크기에 맞춰 합성 레코드를 삭제합니다.
모든 전략은 같은 시그니처 ``(ctx: ReductionContext) -> ReductionResult`` 의 함수이며
:data:`REDUCTION_STRATEGIES` 레지스트리에 등록합니다. 새 전략은 함수 하나를
``@register_strategy("이름")`` 으로 등록하면 됩니다.

전략
- ``random``: 무작위 삭제.
- ``pmse_probability``: ``pmse`` 판별 모델의 확률값(:func:`compute_propensity_scores` 재사용)으로
  합성으로 판별될 확률이 높은(= 유용하지 않은) 레코드부터 삭제.
- ``inference_distance``: ``inference_risk`` (식 4)의 지시함수값을 계산해, 목표값
  (임계값 x ``target_margin``) 안으로 들어오도록 지시함수값 1 / 0 레코드를 비율에 맞춰 삭제.
  1 인 레코드는 ``nearest_distance`` 가 작은(원본에 가까운) 것부터, 0 인 레코드는
  ``nearest_distance`` 가 큰 것부터 삭제합니다. (기존 ``calculate_exact_deletions`` 방식)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdqe import config
from sdqe.metrics.privacy.inference_risk import InferenceDetails, load_or_compute_inference_details
from sdqe.metrics.utility.pmse import compute_propensity_scores
from sdqe.preprocess.validation import Schema

logger = logging.getLogger("sdqe")


@dataclass
class ReductionContext:
    """삭제 전략 입력.

    Attributes:
        original: 원본 (타입 정리 완료).
        synthetic: 삭제 대상 합성 (타입 정리 완료, 0부터 시작하는 행 번호).
        synthetic_full: 합성 파일 전체 (캐시 조회용).
        rows: ``synthetic`` 의 각 행이 ``synthetic_full`` 에서 몇 번째 행인지.
        schema: 분석 스키마.
        target_size: 삭제 후 목표 행 수 (원본 행 수).
        random_seed: 난수 시드.
        options: 전략별 옵션.
    """

    original: pd.DataFrame
    synthetic: pd.DataFrame
    synthetic_full: pd.DataFrame
    rows: np.ndarray
    schema: Schema
    target_size: int
    random_seed: int
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def n_delete(self) -> int:
        """삭제해야 할 행 수."""
        return max(0, len(self.synthetic) - self.target_size)


@dataclass
class ReductionResult:
    """삭제 전략 결과.

    Attributes:
        delete_positions: ``ctx.synthetic`` 기준 삭제할 행 번호 (오름차순).
        details: 결과 JSON 에 기록할 전략별 정보.
        inference_details: ``inference_distance`` 전략이 계산한 레코드별 결과 (캐시 저장용).
    """

    delete_positions: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)
    inference_details: InferenceDetails | None = None


ReductionStrategy = Callable[[ReductionContext], ReductionResult]
REDUCTION_STRATEGIES: dict[str, ReductionStrategy] = {}


def register_strategy(name: str) -> Callable[[ReductionStrategy], ReductionStrategy]:
    """삭제 전략을 레지스트리에 등록하는 데코레이터."""

    def decorator(func: ReductionStrategy) -> ReductionStrategy:
        REDUCTION_STRATEGIES[name] = func
        return func

    return decorator


@register_strategy("random")
def reduce_random(ctx: ReductionContext) -> ReductionResult:
    """무작위로 초과 레코드를 삭제합니다."""
    rng = np.random.default_rng(ctx.random_seed)
    delete = np.sort(rng.choice(len(ctx.synthetic), size=ctx.n_delete, replace=False))
    return ReductionResult(delete, {"signal": "random"})


@register_strategy("pmse_probability")
def reduce_by_pmse_probability(ctx: ReductionContext) -> ReductionResult:
    """합성으로 판별될 확률이 높은 레코드부터 삭제합니다 (동률이면 앞쪽 행 우선)."""
    model = ctx.options.get("model", config.DEFAULT_PMSE_MODEL)
    params = ctx.options.get("model_params")
    scores = compute_propensity_scores(
        ctx.original, ctx.synthetic, ctx.schema, model, params, ctx.random_seed
    )
    prob = scores.synthetic_probability
    order = np.lexsort((np.arange(len(prob)), -prob))
    delete = np.sort(order[: ctx.n_delete])
    return ReductionResult(
        delete,
        {
            "signal": "synthetic_probability",
            "model": model,
            "model_params": params or {},
            "deleted_probability_min": float(prob[delete].min()) if len(delete) else None,
            "kept_probability_max": float(np.delete(prob, delete).max())
            if ctx.n_delete < len(prob)
            else None,
        },
    )


def plan_inference_deletions(
    flags: np.ndarray,
    nearest_distance: np.ndarray,
    target_risk: float,
    target_size: int,
    ties: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """목표 추론위험도에 맞춰 지시함수값 1 / 0 레코드의 삭제 개수와 대상을 정합니다.

    기존 ``calculate_exact_deletions`` 의 계산을 따릅니다. 남은 레코드 중
    ``target_flag1 = round(target_risk * (target_size - 남는 동률 수))`` 개가 1 이 되도록
    1 과 0 을 각각 삭제합니다.

    동률(``d_i == d_or,i``)인 레코드는 추론위험도 계산의 분모에서 빠지므로
    삭제 대상으로 고르지 않고, 비율 계산에서도 제외합니다 (참고문헌 [1] 안내서 68쪽).
    삭제할 개수가 동률이 아닌 레코드 수보다 많을 때만 동률 레코드도 앞에서부터 삭제합니다.

    목표를 정확히 맞출 수 없을 때(한쪽 레코드가 모자람)는 목표에 가까워지는 방향으로만
    삭제합니다. 위험도가 목표보다 낮으면 0 을, 높으면 1 을 먼저 삭제합니다.
    (기존 코드는 1 이 너무 많은 경우에도 0 을 먼저 삭제해 위험도가 오히려 올라갔습니다.)

    Args:
        flags: 레코드별 지시함수값 ``I(d_i < d_or,i)``.
        nearest_distance: 레코드별 ``nearest_distance``.
        target_risk: 목표 추론위험도.
        target_size: 삭제 후 행 수.
        ties: 레코드별 동률 여부. ``None`` 이면 동률이 없다고 봅니다.

    Returns:
        (삭제할 행 번호, 계획 정보).
    """
    flags = np.asarray(flags).astype(bool)
    n = len(flags)
    ties = np.zeros(n, dtype=bool) if ties is None else np.asarray(ties).astype(bool)
    flags = flags & ~ties
    total_delete = max(0, n - target_size)
    idx = np.arange(n)
    ones, zeros, tied = idx[flags], idx[~flags & ~ties], idx[ties]
    cur1, cur0 = len(ones), len(zeros)

    # 동률 레코드는 마지막에만 삭제하므로, 남는 동률 수를 뺀 크기가 비율의 분모입니다.
    delete_tied = max(0, total_delete - (cur1 + cur0))
    denominator = max(0, target_size - (len(tied) - delete_tied))
    target1 = int(round(target_risk * denominator))
    target0 = denominator - target1
    del1, del0 = cur1 - target1, cur0 - target0
    feasible = del1 >= 0 and del0 >= 0
    remaining_delete = total_delete - delete_tied
    if remaining_delete <= 0:
        del1 = del0 = 0
    elif not feasible:
        if del1 < 0:  # 1 이 부족: 위험도가 이미 목표보다 낮음 -> 0 부터 삭제
            del0 = min(cur0, remaining_delete)
            del1 = remaining_delete - del0
        else:  # 0 이 부족: 위험도가 목표보다 높음 -> 1 부터 삭제
            del1 = min(cur1, remaining_delete)
            del0 = remaining_delete - del1

    # 1: 원본에 가장 가까운(거리가 작은) 것부터, 0: 거리가 가장 큰 것부터. 동점이면 앞쪽 행 우선.
    del1_idx = ones[np.lexsort((ones, nearest_distance[ones]))][:del1]
    del0_idx = zeros[np.lexsort((zeros, -nearest_distance[zeros]))][:del0]
    del_tied_idx = tied[:delete_tied]
    delete = np.sort(np.concatenate([del1_idx, del0_idx, del_tied_idx]).astype(np.int64))
    final1 = cur1 - len(del1_idx)
    final_denominator = target_size - (len(tied) - delete_tied)
    info = {
        "target_inference_risk": target_risk,
        "achievable_target": target1 / denominator if denominator else None,
        "target_feasible": feasible,
        "flag1_before": cur1,
        "flag0_before": cur0,
        "n_ties": int(len(tied)),
        "n_ties_deleted": int(delete_tied),
        "flag1_deleted": int(len(del1_idx)),
        "flag0_deleted": int(len(del0_idx)),
        "flag1_deletion_ratio": len(del1_idx) / cur1 if cur1 else 0.0,
        "flag0_deletion_ratio": len(del0_idx) / cur0 if cur0 else 0.0,
        "inference_risk_before": cur1 / (cur1 + cur0) if (cur1 + cur0) else None,
        "inference_risk_after": final1 / final_denominator if final_denominator else None,
    }
    return delete, info


@register_strategy("inference_distance")
def reduce_by_inference_distance(ctx: ReductionContext) -> ReductionResult:
    """추론위험도 거리 비교 결과로 목표값 안에 들도록 레코드를 삭제합니다.

    ``ctx.options`` 에 ``inference_threshold``, ``distance_metric`` 이 필요하며,
    ``original_path``, ``synthetic_path``, ``output_dir`` 가 있으면 캐시를 조회/저장합니다.
    """
    opts = ctx.options
    threshold = opts.get("inference_threshold")
    if threshold is None:
        raise ValueError(
            "inference_distance needs the inference_risk threshold. Run "
            "'sdqe threshold --metric inference_risk' first, or pass --inference_threshold."
        )
    margin = float(opts.get("target_margin", config.DEFAULT_TARGET_MARGIN))
    metric = opts.get("distance_metric", config.DEFAULT_DISTANCE_METRIC)
    full, cache_hit, key = load_or_compute_inference_details(
        ctx.original,
        ctx.synthetic_full,
        ctx.schema,
        metric,
        opts.get("original_path"),
        opts.get("synthetic_path"),
        opts.get("output_dir"),
        opts.get("use_gpu", False),
        opts.get("n_jobs", 1),
        source="preprocess",
    )
    details = full.subset(ctx.rows)
    target = float(threshold) * margin
    delete, info = plan_inference_deletions(
        details.flags, details.nearest_distance, target, ctx.target_size, details.ties
    )
    info.update(
        {
            "signal": "nearest_distance vs reference_distance",
            "distance_metric": metric,
            "inference_threshold": float(threshold),
            "target_margin": margin,
            "cache_hit": cache_hit,
            "cache_key": key,
        }
    )
    return ReductionResult(delete, info, inference_details=details)


def reduce_records(method: str, ctx: ReductionContext) -> ReductionResult:
    """등록된 전략으로 초과 레코드를 삭제할 행을 정합니다.

    Args:
        method: 전략 이름 (:data:`REDUCTION_STRATEGIES` 의 키).
        ctx: 전략 입력.

    Returns:
        :class:`ReductionResult`.
    """
    if method not in REDUCTION_STRATEGIES:
        raise ValueError(
            f"Unknown --reduction_method '{method}'. Use one of {sorted(REDUCTION_STRATEGIES)}."
        )
    result = REDUCTION_STRATEGIES[method](ctx)
    if len(ctx.synthetic) - len(result.delete_positions) != ctx.target_size:
        raise RuntimeError(
            f"Strategy '{method}' produced "
            f"{len(ctx.synthetic) - len(result.delete_positions)} rows; "
            f"expected {ctx.target_size}."
        )
    return result


def output_path_for(output_dir: str | Path) -> Path:
    """전처리된 합성 데이터 저장 경로."""
    return Path(output_dir) / "preprocess" / config.PROCESSED_SYNTHETIC_FILENAME
