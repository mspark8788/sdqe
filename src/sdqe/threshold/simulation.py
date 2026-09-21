"""시뮬레이션 방식 임계값 (참고문헌 [2] 3.1절, 명세 4.2절).

원본을 50:50 으로 무작위 분할해 한쪽을 원본(A), 다른 쪽을 재현(B)으로 가정하고
지표를 계산하는 과정을 반복합니다. 누적된 값의 분위수가 임계값입니다.

반복마다 분할 시드는 ``--random_seed`` 에서 파생시키고 결과에 모두 기록합니다.
분할은 ``A = 순열 앞 floor(n/2) 개``, ``B = 나머지`` 입니다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sdqe import config
from sdqe.common.distance import (
    first_member,
    k_nearest_neighbors,
    nearest_in_subset,
    prepare_features,
    resolve_gpu,
)
from sdqe.metrics.privacy.single_out_risk import record_group_ids, single_out_terms
from sdqe.preprocess.validation import Schema

logger = logging.getLogger("sdqe")


def derive_seeds(random_seed: int, n_simulations: int) -> list[int]:
    """기준 시드에서 반복별 분할 시드를 파생합니다."""
    rng = np.random.default_rng(random_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_simulations)]


def split_halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """원본 행 번호를 A(원본 가정), B(재현 가정)로 나눕니다. 두 배열 모두 오름차순입니다."""
    perm = np.random.default_rng(seed).permutation(n)
    half = n // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def _progress(i: int, total: int, label: str) -> None:
    step = max(1, total // 10)
    if (i + 1) % step == 0 or i + 1 == total:
        logger.info("%s simulation %d/%d", label, i + 1, total)


def simulate_single_out_risk(
    original: pd.DataFrame, schema: Schema, seeds: Sequence[int], duplicate_adjustment: bool
) -> np.ndarray:
    """분할마다 식 (1)/(2) 의 구별위험도 p_hat 을 계산합니다.

    전체 레코드를 한 번만 그룹 번호로 바꾼 뒤, 분할마다 A 의 그룹별 개수를 세어
    B 레코드의 ``d_i`` 를 얻습니다.
    """
    (gid,), n_groups = record_group_ids([original], schema.columns)
    values = np.empty(len(seeds))
    for i, seed in enumerate(seeds):
        a, b = split_halves(len(original), seed)
        counts = np.bincount(gid[a], minlength=n_groups)
        values[i] = single_out_terms(counts[gid[b]], duplicate_adjustment).mean()
        _progress(i, len(seeds), config.SINGLE_OUT_RISK)
    return values


def simulate_inference_risk(
    original: pd.DataFrame,
    schema: Schema,
    seeds: Sequence[int],
    distance_metric: str,
    use_gpu: bool = False,
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """분할마다 식 (4)의 추론위험도를 계산합니다.

    최적화: 원본 전체에서 레코드별 최근접 이웃 목록을 (거리, 행 번호) 순으로 한 번만 만들고,
    분할마다 A 소속 여부로 마스킹만 합니다. 목록 안에 A 소속이 없으면(확률 0.5**K)
    그 레코드만 전수 탐색으로 보완합니다. 거리 계산을 반복하지 않으면서도 분할마다
    전수 탐색한 결과와 정확히 같은 값을 얻습니다(테스트로 검증).

    Returns:
        (반복별 추론위험도, 계산 정보).
    """
    (feats,) = prepare_features(original, [], schema, distance_metric)
    device = resolve_gpu(use_gpu, distance_metric)
    k = min(config.NEIGHBOR_LIST_SIZE, len(original) - 1)
    logger.info(
        "Building %d-nearest-neighbor lists for %d records (computed once).", k, len(original)
    )
    knn_idx, knn_dist = k_nearest_neighbors(feats, k, n_jobs=n_jobs, device=device)
    values = np.empty(len(seeds))
    n_fallback = 0
    n_ties_total = 0
    for i, seed in enumerate(seeds):
        a, b = split_halves(len(original), seed)
        in_a = np.zeros(len(original), dtype=bool)
        in_a[a] = True
        # d_1: B 레코드 -> A 안의 최근접 r_k
        rk, d1, found = first_member(knn_idx[b], knn_dist[b], in_a)
        if not found.all():
            miss = np.flatnonzero(~found)
            rk[miss], d1[miss] = nearest_in_subset(feats, b[miss], a, device=device)
            n_fallback += len(miss)
        # d_2: r_k -> 자기 자신을 제외한 A 안의 최근접 (이웃 목록에는 자기 자신이 없음)
        _, d2, found2 = first_member(knn_idx[rk], knn_dist[rk], in_a)
        if not found2.all():
            miss = np.flatnonzero(~found2)
            _, d2[miss] = nearest_in_subset(feats, rk[miss], a, device=device)
            n_fallback += len(miss)
        # 동률(d_1 == d_2)인 레코드는 분모에서 제외합니다 (참고문헌 [1] 안내서 68쪽).
        n_ties = int((d1 == d2).sum())
        denom = len(b) - n_ties
        values[i] = float((d1 < d2).sum() / denom) if denom else float("nan")
        n_ties_total += n_ties
        _progress(i, len(seeds), config.INFERENCE_RISK)
    return values, {
        "neighbor_list_size": k,
        "n_exhaustive_fallbacks": n_fallback,
        "n_ties_excluded_total": n_ties_total,
    }


def simulate_generic(
    original: pd.DataFrame,
    seeds: Sequence[int],
    metric_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    label: str,
) -> np.ndarray:
    """분할마다 ``metric_fn(A, B)`` 를 계산합니다 (``bivariate_similarity``, ``pmse`` 용)."""
    values = np.empty(len(seeds))
    for i, seed in enumerate(seeds):
        a, b = split_halves(len(original), seed)
        values[i] = metric_fn(
            original.iloc[a].reset_index(drop=True), original.iloc[b].reset_index(drop=True)
        )
        _progress(i, len(seeds), label)
    return values


def size_correction(p_hat: np.ndarray | float) -> np.ndarray | float:
    """식 (5) 크기 보정 ``p_star = 1 - (1 - p_hat)^2``.

    이등분(n/2)으로 계산한 구별위험도를 크기 n 으로 비교할 수 있게 보정합니다.

    References:
        [2] 박민수 외 (2024), 3.1절 식 (5).
    """
    return 1.0 - (1.0 - np.asarray(p_hat, dtype=float)) ** 2
