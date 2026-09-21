"""추론위험도 ``inference_risk`` (참고문헌 [2] 식 (4)).

동일한 레코드가 없더라도 거리가 가까우면 확률적 추론이 가능하다는 점을 측정합니다.

    p_hat = (1/n) * sum_i I( d_i < d_or,i )

    d_i    = nearest_distance   : 재현 레코드 i 와 가장 가까운 원본 레코드 r_k 까지의 거리
    d_or,i = reference_distance : r_k 와 가장 가까운 다른 원본 레코드까지의 거리

완벽한 재현 데이터의 이론값은 0.5 입니다.
기존 Kdata 검증 코드가 ``Originality`` 라고 부르던 값이 바로 이 지표입니다
(같은 값의 옛 이름이며 ``1 - inference_risk`` 가 아닙니다).

구현 메모
- ``d_i == d_or,i`` (동률)인 레코드는 분모 n 에서 제외합니다.
  참고문헌 [1] 안내서 68쪽의 "인 경우의 수만큼 제외하고 나머지 n개의 비율을 계산함" 규칙입니다.
  제외한 개수는 ``n_ties`` 로 보고합니다.
- 최근접 원본이 여러 개면 원본 행 번호가 가장 작은 레코드를 r_k 로 씁니다.
- ``reference_distance`` 는 원본에만 의존하므로 필요한 원본 레코드에 대해 한 번만 계산합니다.
- 계산 결과는 ``results/cache/`` 에 저장하고 같은 입력 해시면 재사용합니다(명세 3.3절).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdqe import config, io_utils
from sdqe.common.distance import nearest_neighbor, prepare_features, resolve_gpu
from sdqe.preprocess.validation import Schema

logger = logging.getLogger("sdqe")


@dataclass
class InferenceDetails:
    """레코드별 추론위험도 계산 결과.

    Attributes:
        nearest_index: 재현 레코드별 최근접 원본 행 번호.
        nearest_distance: 재현 레코드별 최근접 원본 거리 (d_i).
        reference_distance: 그 원본 레코드의 최근접 다른 원본 거리 (d_or,i).
        flags: 지시함수값 ``I(d_i < d_or,i)``.
    """

    nearest_index: np.ndarray
    nearest_distance: np.ndarray
    reference_distance: np.ndarray
    flags: np.ndarray

    @property
    def ties(self) -> np.ndarray:
        """``d_i == d_or,i`` (동률) 여부."""
        return self.nearest_distance == self.reference_distance

    @property
    def n_ties(self) -> int:
        """동률이라 분모에서 제외한 레코드 수."""
        return int(self.ties.sum())

    @property
    def n_evaluated(self) -> int:
        """비율 계산에 쓴 레코드 수 (전체 - 동률)."""
        return len(self.flags) - self.n_ties

    @property
    def value(self) -> float:
        """추론위험도 p_hat = (동률을 뺀 레코드 중 d_i < d_or,i 인 비율)."""
        n = self.n_evaluated
        return float(self.flags.sum() / n) if n else float("nan")

    def subset(self, rows: np.ndarray) -> InferenceDetails:
        """일부 재현 레코드만 남긴 결과.

        레코드별 값은 다른 재현 레코드와 무관하므로 그대로 유효합니다.
        """
        rows = np.asarray(rows)
        return InferenceDetails(
            self.nearest_index[rows],
            self.nearest_distance[rows],
            self.reference_distance[rows],
            self.flags[rows],
        )

    def to_payload(self) -> dict[str, Any]:
        """캐시 저장용 딕셔너리."""
        return {
            "inference_risk": self.value,
            "n_ties": self.n_ties,
            "n_evaluated": self.n_evaluated,
            "nearest_index": self.nearest_index,
            "nearest_distance": self.nearest_distance,
            "reference_distance": self.reference_distance,
            "flags": self.flags,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> InferenceDetails:
        """캐시 딕셔너리에서 복원합니다."""
        return cls(
            nearest_index=np.asarray(payload["nearest_index"], dtype=np.int64),
            nearest_distance=np.asarray(payload["nearest_distance"], dtype=np.float64),
            reference_distance=np.asarray(payload["reference_distance"], dtype=np.float64),
            flags=np.asarray(payload["flags"], dtype=np.int8),
        )


def compute_inference_details(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
    distance_metric: str = config.DEFAULT_DISTANCE_METRIC,
    use_gpu: bool = False,
    n_jobs: int = 1,
) -> InferenceDetails:
    """재현 레코드별로 식 (4)의 거리와 지시함수값을 계산합니다.

    Args:
        original: 원본 데이터 (타입 정리 완료).
        synthetic: 합성 데이터 (타입 정리 완료, 원본과 같은 열 순서).
        schema: 분석 스키마.
        distance_metric: ``gower`` 또는 ``cosine``.
        use_gpu: cosine 수치형 내적에 GPU 사용.
        n_jobs: 청크 병렬 처리 스레드 수.

    Returns:
        :class:`InferenceDetails`.

    References:
        [2] 박민수 외 (2024), 2.3절 식 (4).
    """
    if len(original) < 2:
        raise ValueError("inference_risk requires at least two original records.")
    feats_orig, feats_syn = prepare_features(original, [synthetic], schema, distance_metric)
    device = resolve_gpu(use_gpu, distance_metric)

    # d_i: 재현 레코드 -> 최근접 원본
    nearest_idx, nearest_dist = nearest_neighbor(
        feats_syn, feats_orig, n_jobs=n_jobs, device=device
    )
    # d_or: 최근접으로 선택된 원본 레코드 -> 자기 자신을 제외한 최근접 원본 (필요한 원본만 1회 계산)
    unique_idx = np.unique(nearest_idx)
    _, ref_dist_unique = nearest_neighbor(
        feats_orig.take(unique_idx), feats_orig, exclude=unique_idx, n_jobs=n_jobs, device=device
    )
    reference_dist = ref_dist_unique[np.searchsorted(unique_idx, nearest_idx)]
    flags = (nearest_dist < reference_dist).astype(np.int8)
    return InferenceDetails(nearest_idx, nearest_dist, reference_dist, flags)


# ---------------------------------------------------------------------------
# 캐시 연동 (명세 3.3절)
# ---------------------------------------------------------------------------
def inference_cache_params(schema: Schema) -> dict[str, Any]:
    """캐시 키에 넣을 관련 파라미터."""
    return {
        "metric": config.INFERENCE_RISK,
        "schema": schema.to_dict(),
        "distance_decimals": config.DISTANCE_DECIMALS,
    }


def inference_cache_key(
    original_path: str | Path, synthetic_path: str | Path, schema: Schema, distance_metric: str
) -> str:
    """추론위험도 캐시 키. 전처리와 지표 계산이 이 함수를 함께 씁니다."""
    return io_utils.make_cache_key(
        original_path, synthetic_path, distance_metric, inference_cache_params(schema)
    )


def store_inference_cache(
    output_dir: str | Path,
    key: str,
    details: InferenceDetails,
    distance_metric: str,
    source: str,
) -> Path:
    """추론위험도 거리 결과를 캐시에 저장합니다."""
    path = io_utils.save_cache(
        output_dir,
        key,
        {
            "metric": config.INFERENCE_RISK,
            "distance_metric": distance_metric,
            "source": source,
            **details.to_payload(),
        },
    )
    logger.info("Saved inference_risk distance cache: %s", path)
    return path


def load_or_compute_inference_details(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
    distance_metric: str,
    original_path: str | Path | None,
    synthetic_path: str | Path | None,
    output_dir: str | Path | None,
    use_gpu: bool = False,
    n_jobs: int = 1,
    source: str = "metric",
) -> tuple[InferenceDetails, bool, str | None]:
    """캐시를 먼저 조회하고, 없을 때만 새로 계산해 저장합니다.

    Args:
        original: 원본 데이터.
        synthetic: 합성 데이터.
        schema: 분석 스키마.
        distance_metric: 거리 측정 방식.
        original_path: 원본 파일 경로 (없으면 캐시를 쓰지 않음).
        synthetic_path: 합성 파일 경로 (없으면 캐시를 쓰지 않음).
        output_dir: 결과 최상위 디렉터리 (없으면 캐시를 쓰지 않음).
        use_gpu: cosine GPU 사용.
        n_jobs: 병렬 스레드 수.
        source: 캐시 생성 단계 이름 (기록용).

    Returns:
        (계산 결과, 캐시 사용 여부, 캐시 키).
    """
    key = None
    if original_path and synthetic_path and output_dir:
        key = inference_cache_key(original_path, synthetic_path, schema, distance_metric)
        cached = io_utils.load_cache(output_dir, key)
        if cached is not None and len(cached.get("flags", [])) == len(synthetic):
            logger.info("Using cached inference_risk distances (cache key %s).", key)
            return InferenceDetails.from_payload(cached), True, key
    details = compute_inference_details(
        original, synthetic, schema, distance_metric, use_gpu, n_jobs
    )
    if key is not None:
        store_inference_cache(output_dir, key, details, distance_metric, source)
    return details, False, key


def interpret_inference_risk(value: float) -> str:
    """이론 기준값 0.5 대비 해석 문구."""
    ref = config.INFERENCE_REFERENCE_VALUE
    if value > ref:
        return f"above the theoretical reference {ref}: inference risk is on the high side"
    if value < ref:
        return f"below the theoretical reference {ref}: safer, but utility may be lower"
    return f"equal to the theoretical reference {ref}: balanced"


def summarize_inference(details: InferenceDetails) -> dict[str, Any]:
    """결과 JSON 에 넣을 요약."""
    value = details.value
    return {
        "value": value,
        "reference_value": config.INFERENCE_REFERENCE_VALUE,
        "interpretation": interpret_inference_risk(value),
        "n_flagged": int(details.flags.sum()),
        "n_evaluated": details.n_evaluated,
        "n_ties": details.n_ties,
        "tie_policy": "records with d_i == d_or,i are excluded from the denominator "
        "(reference [1], p. 68)",
        "nearest_distance_summary": _describe(details.nearest_distance),
        "reference_distance_summary": _describe(details.reference_distance),
    }


def _describe(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {}
    return {
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }
