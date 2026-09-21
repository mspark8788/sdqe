"""분포가정 방식 임계값 (참고문헌 [2] 3.2절, 명세 4.3절).

이등분 없이 원본 전체로 한 번 계산한 비율 p 에 대해 이항비율의 정규근사
``p_hat ~ N(p, p(1-p)/n)`` 를 가정하고, 그 분위수를 임계값으로 씁니다.
이등분하지 않으므로 식 (5) 보정은 적용하지 않습니다.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from sdqe import config
from sdqe.common.distance import k_nearest_neighbors, prepare_features, resolve_gpu
from sdqe.metrics.privacy.single_out_risk import leave_one_out_duplicate_counts, single_out_terms
from sdqe.preprocess.validation import Schema

logger = logging.getLogger("sdqe")


def single_out_risk_proportion(
    original: pd.DataFrame, schema: Schema, duplicate_adjustment: bool
) -> float:
    """각 원본 레코드를 재현 레코드로 보고 나머지 원본과 비교한 식 (1)/(2) 의 p.

    ``d_i`` 는 자기 자신을 뺀 나머지 원본에서 동일한 레코드 수입니다.
    """
    d = leave_one_out_duplicate_counts(original, schema)
    return float(single_out_terms(d, duplicate_adjustment).mean())


def inference_risk_proportion(
    original: pd.DataFrame,
    schema: Schema,
    distance_metric: str,
    use_gpu: bool = False,
    n_jobs: int = 1,
) -> tuple[float, dict[str, Any]]:
    """원본만으로 식 (4) 의 비율 p 를 계산합니다.

    레코드 r_1 과 가장 가까운 레코드 r_k 사이 거리 d_1, r_1 을 제외하고 r_k 와 가장 가까운
    레코드까지 거리 d_2 를 구해 ``d_1 < d_2`` 이면 1 로 집계합니다.
    최근접 레코드가 여러 개면 행 번호가 가장 작은 레코드를 r_k 로 씁니다.
    ``d_1 == d_2`` 인 레코드는 분모에서 제외합니다 (참고문헌 [1] 안내서 68쪽).

    Returns:
        (p, 계산 정보). 계산 정보의 ``n_evaluated`` 가 정규근사에 쓸 n 입니다.
    """
    if len(original) < 3:
        raise ValueError("inference_risk distribution method needs at least three records.")
    (feats,) = prepare_features(original, [], schema, distance_metric)
    device = resolve_gpu(use_gpu, distance_metric)
    # 자기 자신을 제외한 최근접 2개면 충분: r_k 의 목록에서 r_1 이 아닌 첫 레코드가 d_2
    idx, dist = k_nearest_neighbors(feats, 2, n_jobs=n_jobs, device=device)
    rows = np.arange(len(original))
    rk = idx[:, 0]
    d1 = dist[:, 0]
    d2 = np.where(idx[rk, 0] != rows, dist[rk, 0], dist[rk, 1])
    n_ties = int((d1 == d2).sum())
    n_evaluated = len(original) - n_ties
    if n_evaluated == 0:
        raise ValueError("All records are ties (d_1 == d_2); inference_risk cannot be computed.")
    p = float((d1 < d2).sum() / n_evaluated)
    return p, {"n_ties": n_ties, "n_evaluated": n_evaluated}


def normal_quantile_threshold(p: float, n: int, quantile: float) -> dict[str, Any]:
    """``N(p, p(1-p)/n)`` 의 분위수를 임계값으로 계산합니다.

    ``n*p*(1-p)`` 가 작으면 근사가 부정확하므로 경고하고 시뮬레이션 방식을 권고합니다.

    Returns:
        ``threshold``, ``mean``, ``variance``, ``std``, ``n``, ``npq`` 를 담은 딕셔너리.
    """
    variance = p * (1.0 - p) / n
    std = math.sqrt(variance)
    threshold = float(norm.ppf(quantile, loc=p, scale=std)) if std > 0 else float(p)
    npq = n * p * (1.0 - p)
    if npq < config.NORMAL_APPROX_MIN_NPQ:
        logger.warning(
            "n*p*(1-p) = %.3f is below %.0f: the normal approximation may be inaccurate. "
            "Consider --method simulation.",
            npq,
            config.NORMAL_APPROX_MIN_NPQ,
        )
    return {
        "threshold": threshold,
        "mean": p,
        "variance": variance,
        "std": std,
        "n": n,
        "npq": npq,
        "normal_approximation_reliable": npq >= config.NORMAL_APPROX_MIN_NPQ,
    }
