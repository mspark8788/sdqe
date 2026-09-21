"""Gower / Cosine 거리 계산 (공용, 명세 4.7절).

임계값 계산과 ``inference_risk`` 계산이 모두 이 모듈의 함수를 호출합니다.

거리 정의
---------
``gower`` (참고문헌 [6] Gower, 1971)
    범주형은 불일치 여부(0/1), 수치형은 원본 min/max 범위로 나눈 절대차.
    ``d(a, b) = (sum_num |a_j - b_j| / range_j + sum_cat I(a_c != b_c)) / p``.
    원본 범위가 0인 수치형 열은 0으로 기여합니다. 수치형 결측은 Gower 원 정의대로
    해당 변수를 분자와 분모에서 함께 제외합니다.

``cosine`` (기존 Kdata 검증 코드의 혼합 방식 유지)
    범주형 유사도 ``2 * (일치 비율) - 1``, 수치형 유사도는 원본 기준 표준화 후 코사인 유사도.
    두 유사도를 열 개수로 가중평균한 뒤 ``1 - 유사도`` 를 거리로 씁니다.
    ``d(a, b) = 1 - (2 * matches - p_cat + p_num * cos(a_num, b_num)) / p``.
    수치형 벡터 노름이 0이면 코사인 유사도를 0으로 둡니다.

공통 규칙
---------
- 거리값은 ``config.DISTANCE_DECIMALS`` 자리로 반올림합니다. 수학적으로 같은 거리가
  부동소수점 잡음 때문에 다르게 판정되지 않게 하기 위함입니다.
- 최근접 레코드가 여러 개(동률)이면 대상 데이터에서 행 번호가 가장 작은 레코드를 고릅니다.
- 거리 행렬은 청크 단위로 계산하여 메모리 사용량을 제한합니다.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sdqe import config
from sdqe.preprocess.validation import Schema

logger = logging.getLogger("sdqe")

# 범주형 수준 총 개수가 이 값 이하이면 one-hot 행렬곱으로 일치 수를 셉니다(정수 연산이라 결과 동일).
_ONEHOT_MAX_LEVELS = 1024


@dataclass(frozen=True)
class DistanceFeatures:
    """거리 계산용으로 변환된 데이터.

    Attributes:
        metric: ``gower`` 또는 ``cosine``.
        numeric: 수치형 특성 (n, p_num). gower 는 범위로 나눈 값, cosine 은 단위 노름 벡터.
        categorical: 범주형 코드 (n, p_cat). 결측은 별도 코드 하나로 취급.
        onehot: 범주형 one-hot (n, L). 수준 수가 많으면 ``None``.
        n_features: 전체 변수 개수 p.
    """

    metric: str
    numeric: np.ndarray
    categorical: np.ndarray
    onehot: np.ndarray | None
    n_features: int

    @property
    def n_rows(self) -> int:
        """행 개수."""
        return int(self.numeric.shape[0])

    @property
    def has_missing_numeric(self) -> bool:
        """수치형 결측 존재 여부."""
        return bool(self.numeric.size) and bool(np.isnan(self.numeric).any())

    def take(self, rows: np.ndarray) -> DistanceFeatures:
        """지정한 행만 담은 새 객체를 반환합니다."""
        rows = np.asarray(rows)
        return DistanceFeatures(
            metric=self.metric,
            numeric=self.numeric[rows],
            categorical=self.categorical[rows],
            onehot=None if self.onehot is None else self.onehot[rows],
            n_features=self.n_features,
        )


# ---------------------------------------------------------------------------
# 특성 변환
# ---------------------------------------------------------------------------
def prepare_features(
    reference: pd.DataFrame,
    others: Sequence[pd.DataFrame],
    schema: Schema,
    metric: str,
) -> list[DistanceFeatures]:
    """원본(reference) 기준으로 스케일링 기준을 정하고 모든 데이터를 같은 공간으로 변환합니다.

    스케일링 기준(gower 의 min/max, cosine 의 평균/표준편차)은 원본에서만 구해
    합성에도 같은 기준을 적용합니다. 따라서 원본-원본 거리는 원본에만 의존합니다.

    Args:
        reference: 원본 데이터 (타입 정리 완료).
        others: 같은 공간으로 변환할 다른 데이터들 (예: 합성 데이터).
        schema: 분석 스키마.
        metric: ``gower`` 또는 ``cosine``.

    Returns:
        ``[reference 변환 결과, *others 변환 결과]``.

    References:
        [6] Gower, J. C. (1971). A General Coefficient of Similarity and Some of Its Properties.
    """
    if metric not in config.DISTANCE_METRICS:
        raise ValueError(
            f"Unsupported distance_metric '{metric}'. Use one of {config.DISTANCE_METRICS}."
        )
    frames = [reference, *others]
    num_cols = list(schema.numeric)
    cat_cols = list(schema.categorical)
    p = len(num_cols) + len(cat_cols)
    if p == 0:
        raise ValueError("No columns available for distance computation.")

    ref_num = (
        reference[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(reference), 0))
    )
    if metric == "cosine" and num_cols and np.isnan(ref_num).any():
        raise ValueError(
            "cosine distance does not support missing numeric values. "
            "Impute them or use --distance_metric gower."
        )
    if metric == "gower":
        lo = np.nanmin(ref_num, axis=0) if num_cols else np.empty(0)
        hi = np.nanmax(ref_num, axis=0) if num_cols else np.empty(0)
        scale = hi - lo
    else:
        # sklearn StandardScaler 와 같은 규칙: 모집단 표준편차, 0이면 1로 대체
        lo = ref_num.mean(axis=0) if num_cols else np.empty(0)
        scale = ref_num.std(axis=0) if num_cols else np.empty(0)
    scale = np.where(scale > 0, scale, 1.0 if metric == "cosine" else np.inf)

    # 범주형 코드: 모든 데이터의 수준을 합쳐 같은 코드 체계를 씁니다. 결측은 코드 0.
    codes_per_frame: list[np.ndarray] = [
        np.zeros((len(f), len(cat_cols)), dtype=np.int64) for f in frames
    ]
    level_counts = []
    for j, col in enumerate(cat_cols):
        levels = pd.Index(
            pd.unique(pd.concat([f[col] for f in frames], ignore_index=True).dropna())
        )
        level_counts.append(len(levels) + 1)
        for k, f in enumerate(frames):
            idx = levels.get_indexer(f[col])
            codes_per_frame[k][:, j] = idx + 1  # -1(결측) -> 0
    total_levels = int(sum(level_counts))
    use_onehot = bool(cat_cols) and total_levels <= _ONEHOT_MAX_LEVELS
    offsets = (
        np.concatenate([[0], np.cumsum(level_counts)[:-1]]).astype(np.int64) if cat_cols else None
    )

    result = []
    for f, codes in zip(frames, codes_per_frame):
        num = f[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(f), 0))
        if metric == "cosine" and num_cols and np.isnan(num).any():
            raise ValueError("cosine distance does not support missing numeric values.")
        if metric == "gower":
            # 범위 0(상수열)은 scale=inf -> 0 기여
            num = np.where(np.isnan(num), np.nan, (num - lo) / scale) if num_cols else num
            num = np.where(np.isinf(scale), 0.0, num) if num_cols else num
        else:
            num = (num - lo) / scale if num_cols else num
            norms = np.linalg.norm(num, axis=1, keepdims=True) if num_cols else None
            if num_cols:
                num = np.divide(num, norms, out=np.zeros_like(num), where=norms > 0)
        onehot = None
        if use_onehot:
            onehot = np.zeros((len(f), total_levels), dtype=np.float64)
            rows = np.repeat(np.arange(len(f)), len(cat_cols))
            onehot[rows, (codes + offsets).ravel()] = 1.0
        result.append(
            DistanceFeatures(
                metric=metric,
                numeric=np.ascontiguousarray(num, dtype=np.float64),
                categorical=np.ascontiguousarray(codes),
                onehot=onehot,
                n_features=p,
            )
        )
    return result


# ---------------------------------------------------------------------------
# GPU (cosine 전용)
# ---------------------------------------------------------------------------
def resolve_gpu(use_gpu: bool, metric: str) -> Any | None:
    """GPU 사용 가능 여부를 확인해 torch device 를 반환합니다.

    torch 또는 CUDA 가 없으면 경고 후 ``None``(CPU)을 반환하며 실행을 중단하지 않습니다.

    Args:
        use_gpu: ``--use_gpu`` 값.
        metric: 거리 측정 방식. GPU 는 ``cosine`` 의 수치형 내적에만 씁니다.

    Returns:
        torch device 또는 ``None``.
    """
    if not use_gpu:
        return None
    if metric != "cosine":
        logger.warning("--use_gpu applies to cosine distance only; computing %s on CPU.", metric)
        return None
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        logger.warning("--use_gpu requested but torch is not installed; falling back to CPU.")
        return None
    if not torch.cuda.is_available():
        logger.warning("--use_gpu requested but CUDA is not available; falling back to CPU.")
        return None
    return torch.device("cuda")


def _cosine_numeric_block(q: np.ndarray, t: np.ndarray, device: Any | None) -> np.ndarray:
    if device is None:
        return q @ t.T
    import torch  # noqa: PLC0415

    qt = torch.as_tensor(q, dtype=torch.float64, device=device)
    tt = torch.as_tensor(t, dtype=torch.float64, device=device)
    return (qt @ tt.T).cpu().numpy()


# ---------------------------------------------------------------------------
# 블록 거리
# ---------------------------------------------------------------------------
def _match_counts(q: DistanceFeatures, t: DistanceFeatures) -> np.ndarray:
    """범주형 일치 개수 (nq, nt). 정수 연산이므로 두 경로의 결과가 같습니다."""
    if q.onehot is not None and t.onehot is not None:
        return q.onehot @ t.onehot.T
    counts = np.zeros((q.n_rows, t.n_rows), dtype=np.float64)
    for c in range(q.categorical.shape[1]):
        counts += q.categorical[:, c, None] == t.categorical[None, :, c]
    return counts


def distance_block(
    q: DistanceFeatures, t: DistanceFeatures, device: Any | None = None
) -> np.ndarray:
    """두 데이터 사이의 거리 행렬 (nq, nt) 을 계산합니다.

    Args:
        q: 질의 데이터 특성.
        t: 대상 데이터 특성.
        device: cosine 수치형 내적에 쓸 torch device (없으면 CPU).

    Returns:
        반올림된 거리 행렬.
    """
    p = q.n_features
    p_num = q.numeric.shape[1]
    p_cat = q.categorical.shape[1]
    if q.metric == "gower":
        acc = np.zeros((q.n_rows, t.n_rows), dtype=np.float64)
        denom: np.ndarray | float = float(p)
        if q.has_missing_numeric or t.has_missing_numeric:
            valid_cnt = np.zeros_like(acc)
            for j in range(p_num):
                diff = np.abs(q.numeric[:, j, None] - t.numeric[None, :, j])
                valid = ~np.isnan(diff)
                acc += np.where(valid, diff, 0.0)
                valid_cnt += valid
            denom = valid_cnt + p_cat
        else:
            for j in range(p_num):
                acc += np.abs(q.numeric[:, j, None] - t.numeric[None, :, j])
        if p_cat:
            acc += p_cat - _match_counts(q, t)
        with np.errstate(invalid="ignore", divide="ignore"):
            acc /= denom
        acc[~np.isfinite(acc)] = np.inf
    else:
        sim = np.zeros((q.n_rows, t.n_rows), dtype=np.float64)
        if p_cat:
            sim += 2.0 * _match_counts(q, t) - p_cat
        if p_num:
            sim += p_num * _cosine_numeric_block(q.numeric, t.numeric, device)
        acc = 1.0 - sim / p
    np.round(acc, config.DISTANCE_DECIMALS, out=acc)
    return acc


def pairwise_distances(
    a: DistanceFeatures, b: DistanceFeatures, device: Any | None = None
) -> np.ndarray:
    """전체 거리 행렬을 한 번에 계산합니다. 소규모 데이터와 테스트용입니다."""
    return distance_block(a, b, device)


def _chunk_rows(n_target: int) -> int:
    per_row = 8 * max(n_target, 1) * 4  # 거리 행렬 + 임시 배열 여유분
    return int(max(1, min(4096, config.DISTANCE_CHUNK_BYTES // per_row)))


def _resolve_jobs(n_jobs: int) -> int:
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


def _run_chunks(func, n_rows: int, chunk: int, n_jobs: int) -> None:
    starts = range(0, n_rows, chunk)
    jobs = _resolve_jobs(n_jobs)
    if jobs == 1 or n_rows <= chunk:
        for s in starts:
            func(s, min(s + chunk, n_rows))
    else:
        Parallel(n_jobs=jobs, prefer="threads")(
            delayed(func)(s, min(s + chunk, n_rows)) for s in starts
        )


# ---------------------------------------------------------------------------
# 최근접 이웃
# ---------------------------------------------------------------------------
def nearest_neighbor(
    query: DistanceFeatures,
    target: DistanceFeatures,
    exclude: np.ndarray | None = None,
    n_jobs: int = 1,
    device: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """질의 레코드마다 대상 데이터에서 가장 가까운 레코드를 찾습니다.

    동률이면 대상 데이터의 행 번호가 가장 작은 레코드를 고릅니다.

    Args:
        query: 질의 데이터 특성 (nq 행).
        target: 대상 데이터 특성 (nt 행).
        exclude: 질의별로 제외할 대상 행 번호 (nq,). -1 이면 제외 없음.
            원본-원본 비교에서 자기 자신을 제외할 때 씁니다.
        n_jobs: 청크 병렬 처리 스레드 수.
        device: cosine GPU device.

    Returns:
        (최근접 대상 행 번호 (nq,), 거리 (nq,)).
    """
    nq = query.n_rows
    idx_out = np.empty(nq, dtype=np.int64)
    dist_out = np.empty(nq, dtype=np.float64)
    chunk = _chunk_rows(target.n_rows)

    def work(s: int, e: int) -> None:
        block = distance_block(query.take(np.arange(s, e)), target, device)
        if exclude is not None:
            ex = exclude[s:e]
            rows = np.flatnonzero(ex >= 0)
            block[rows, ex[rows]] = np.inf
        best = np.argmin(block, axis=1)
        idx_out[s:e] = best
        dist_out[s:e] = block[np.arange(e - s), best]

    _run_chunks(work, nq, chunk, n_jobs)
    return idx_out, dist_out


def k_nearest_neighbors(
    features: DistanceFeatures,
    k: int,
    n_jobs: int = 1,
    device: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """같은 데이터 안에서 각 레코드의 최근접 이웃 k개를 찾습니다 (자기 자신 제외).

    이웃 목록은 (거리, 행 번호) 사전식 순서로 정렬됩니다. 따라서 임의의 부분집합 S 에 대해
    "목록에서 처음 나오는 S 소속 레코드"는 "S 안에서 거리가 가장 작고 행 번호가 가장 작은
    레코드"와 정확히 같습니다. 시뮬레이션은 이 성질로 분할마다 거리를 다시 계산하지 않습니다.

    Args:
        features: 데이터 특성 (n 행).
        k: 이웃 수. ``n - 1`` 을 넘으면 ``n - 1`` 로 줄입니다.
        n_jobs: 청크 병렬 처리 스레드 수.
        device: cosine GPU device.

    Returns:
        (이웃 행 번호 (n, k), 거리 (n, k)).
    """
    n = features.n_rows
    k = int(min(k, n - 1))
    if k < 1:
        raise ValueError("At least two records are required to search neighbors.")
    idx_out = np.empty((n, k), dtype=np.int64)
    dist_out = np.empty((n, k), dtype=np.float64)
    chunk = _chunk_rows(n)

    def work(s: int, e: int) -> None:
        block = distance_block(features.take(np.arange(s, e)), features, device)
        block[np.arange(e - s), np.arange(s, e)] = np.inf
        kth = np.partition(block, k - 1, axis=1)[:, k - 1]
        for r in range(e - s):
            row = block[r]
            less = np.flatnonzero(row < kth[r])
            equal = np.flatnonzero(row == kth[r])[: k - len(less)]
            cand = np.concatenate([less, equal])
            order = np.lexsort((cand, row[cand]))
            idx_out[s + r] = cand[order]
            dist_out[s + r] = row[cand[order]]

    _run_chunks(work, n, chunk, n_jobs)
    return idx_out, dist_out


def first_member(
    neighbor_idx: np.ndarray,
    neighbor_dist: np.ndarray,
    member_mask: np.ndarray,
    skip: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """이웃 목록에서 처음 나오는 집합 소속 레코드를 찾습니다.

    Args:
        neighbor_idx: 이웃 행 번호 (m, k) — :func:`k_nearest_neighbors` 결과의 일부 행.
        neighbor_dist: 이웃 거리 (m, k).
        member_mask: 전체 레코드에 대한 집합 소속 여부 (n,).
        skip: 행마다 추가로 건너뛸 레코드 번호 (m,). -1 이면 없음.

    Returns:
        (찾은 행 번호 (m,), 거리 (m,), 찾았는지 여부 (m,)).
        목록 안에 소속 레코드가 없으면 찾았는지 여부가 False 이며,
        호출자가 전수 탐색으로 보완합니다.
    """
    member = member_mask[neighbor_idx]
    if skip is not None:
        member &= neighbor_idx != skip[:, None]
    found = member.any(axis=1)
    pos = np.argmax(member, axis=1)
    rows = np.arange(len(neighbor_idx))
    return neighbor_idx[rows, pos], neighbor_dist[rows, pos], found


def nearest_in_subset(
    features: DistanceFeatures,
    query_rows: np.ndarray,
    subset_rows: np.ndarray,
    skip: np.ndarray | None = None,
    device: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """전수 탐색으로 부분집합 안의 최근접 레코드를 찾습니다 (이웃 목록 보완용).

    Args:
        features: 전체 데이터 특성.
        query_rows: 질의 행 번호.
        subset_rows: 대상 부분집합 행 번호 (오름차순 정렬 필요, 동률 시 작은 번호 우선).
        skip: 질의별로 제외할 전체 기준 행 번호 (-1 이면 없음). 질의 자신은 항상 제외합니다.
        device: cosine GPU device.

    Returns:
        (전체 기준 행 번호, 거리).
    """
    subset_rows = np.sort(np.asarray(subset_rows))
    block = distance_block(features.take(query_rows), features.take(subset_rows), device)
    pos_of = {int(r): i for i, r in enumerate(subset_rows)}
    for r, q in enumerate(query_rows):
        for excluded in (q, -1 if skip is None else skip[r]):
            i = pos_of.get(int(excluded))
            if i is not None:
                block[r, i] = np.inf
    best = np.argmin(block, axis=1)
    return subset_rows[best], block[np.arange(len(query_rows)), best]
