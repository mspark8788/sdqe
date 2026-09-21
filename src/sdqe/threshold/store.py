"""임계값 결과 파일 위치 조회와 판정.

지표 계산과 전처리(``inference_distance``)가 같은 조회/판정 함수를 씁니다.
"""

from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Any

from sdqe import config, io_utils

logger = logging.getLogger("sdqe")

_OPERATORS = {"<": operator.lt, "<=": operator.le}


def threshold_path(output_dir: str | Path, metric: str) -> Path:
    """``output_dir/threshold/<metric>.json`` 경로."""
    return Path(output_dir) / "threshold" / f"{metric}.json"


def load_threshold_record(
    output_dir: str | Path | None, metric: str, threshold_file: str | Path | None = None
) -> tuple[dict[str, Any] | None, Path | None]:
    """임계값 결과 JSON 을 읽습니다.

    Args:
        output_dir: 결과 최상위 디렉터리.
        metric: 지표 식별자.
        threshold_file: 명시한 임계값 파일 경로. 있으면 이것을 우선합니다.

    Returns:
        (임계값 기록, 파일 경로). 파일이 없으면 (None, None).

    Raises:
        FileNotFoundError: 명시한 파일이 없을 때.
        ValueError: 파일의 지표가 요청한 지표와 다를 때.
    """
    if threshold_file:
        path = Path(threshold_file)
        if not path.is_file():
            raise FileNotFoundError(f"Threshold file not found: {path}")
    elif output_dir is not None:
        path = threshold_path(output_dir, metric)
        if not path.is_file():
            return None, None
    else:
        return None, None
    record = io_utils.load_json(path)
    if record.get("metric") != metric:
        raise ValueError(f"Threshold file {path} is for '{record.get('metric')}', not '{metric}'.")
    return record, path


def check_distance_metric(record: dict[str, Any], distance_metric: str, path: Path | None) -> None:
    """지표와 임계값의 ``distance_metric`` 이 같은지 확인합니다 (명세 4.7절).

    Raises:
        ValueError: 서로 다를 때. 다른 거리로 계산한 값끼리의 비교는 의미가 없습니다.
    """
    used = record.get("parameters", {}).get("distance_metric")
    if record.get("threshold_source") != config.THRESHOLD_SOURCE_USER and used != distance_metric:
        raise ValueError(
            f"distance_metric mismatch: metric uses '{distance_metric}' but threshold {path} "
            f"was computed with '{used}'. Recompute one of them with the same --distance_metric."
        )


def judge(
    metric: str,
    value: float | None,
    record: dict[str, Any] | None,
    path: Path | None,
    metric_parameters: dict[str, Any],
) -> dict[str, Any]:
    """지표값을 임계값과 비교합니다.

    Args:
        metric: 지표 식별자.
        value: 지표값.
        record: 임계값 기록 (없으면 판정하지 않음).
        path: 임계값 파일 경로.
        metric_parameters: 지표 계산에 쓴 파라미터 (정합성 확인용).

    Returns:
        ``threshold``, ``threshold_source``, ``threshold_file``, ``passed``, ``judge_warnings``.

    Raises:
        ValueError: ``inference_risk`` 의 ``distance_metric`` 이 다를 때.
    """
    out: dict[str, Any] = {
        "threshold": None,
        "threshold_source": None,
        "threshold_file": None,
        "passed": None,
        "judge_warnings": [],
    }
    if record is None:
        logger.info("No threshold found for %s; skipping pass/fail judgement.", metric)
        return out
    warnings: list[str] = []
    params = record.get("parameters", {})
    if metric == config.INFERENCE_RISK:
        check_distance_metric(record, metric_parameters["distance_metric"], path)
    if (
        metric == config.SINGLE_OUT_RISK
        and record.get("threshold_source") != config.THRESHOLD_SOURCE_USER
    ):
        if (
            record.get("method") == config.METHOD_SIMULATION
            and params.get("size_correction") != "on"
        ):
            warnings.append(
                "threshold was simulated without size correction (equation 5); compare against "
                "a random half of the original data or recompute with --size_correction on"
            )
        if params.get("duplicate_adjustment") != metric_parameters.get("duplicate_adjustment"):
            warnings.append(
                "duplicate_adjustment differs between the metric and the threshold "
                "(equation 1 vs 2); the comparison is not meaningful"
            )
    if metric == config.PMSE and record.get("threshold_source") != config.THRESHOLD_SOURCE_USER:
        for key in ("model", "model_params"):
            if params.get(key) != metric_parameters.get(key):
                warnings.append(f"pmse {key} differs between the metric and the threshold")
    for w in warnings:
        logger.warning("%s: %s", metric, w)
    threshold = record.get("threshold")
    passed = None
    if value is not None and threshold is not None:
        passed = bool(_OPERATORS[config.PASS_RULES[metric]](value, threshold))
    out.update(
        {
            "threshold": threshold,
            "threshold_source": record.get("threshold_source"),
            "threshold_file": path.as_posix() if path else None,
            "pass_rule": f"value {config.PASS_RULES[metric]} threshold",
            "passed": passed,
            "judge_warnings": warnings,
        }
    )
    return out
