"""통합 실행 (``sdqe report``).

전처리부터 모든 지표까지 실행하고 ``output_dir/report.json`` 으로 통합 리포트를 만듭니다.

실행 순서
1. 임계값 계산 (원본만 필요하므로 먼저 계산합니다. ``inference_distance`` 전처리가
   추론위험도 임계값을 쓰기 때문입니다.)
2. 전처리 (정합성 검증, 초과 레코드 삭제)
3. 지표 계산 (전처리된 합성 데이터 기준)
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sdqe import config, io_utils
from sdqe.metrics.runner import MetricOptions, run_metrics
from sdqe.preprocess.pipeline import PreprocessOptions, run_preprocess
from sdqe.threshold.runner import ThresholdOptions, run_threshold

logger = logging.getLogger("sdqe")


def build_options(cls: type, values: dict[str, Any], **overrides: Any) -> Any:
    """``values`` 중 dataclass 필드에 해당하는 값만 골라 옵션 객체를 만듭니다."""
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in values.items() if k in names and v is not None}
    kwargs.update(overrides)
    return cls(**kwargs)


def select_metrics(
    metrics: Sequence[str] | None, skip: Sequence[str] | None, universe: Sequence[str]
) -> list[str]:
    """``--metrics`` / ``--skip_metrics`` 로 계산할 지표를 고릅니다."""
    chosen = list(metrics) if metrics else list(universe)
    bad = [m for m in chosen + list(skip or []) if m not in universe]
    if bad:
        raise ValueError(f"Unknown or out-of-group metrics {bad}. Allowed: {list(universe)}")
    return [m for m in chosen if m not in set(skip or [])]


def run_report(values: dict[str, Any]) -> dict[str, Any]:
    """전체 파이프라인을 실행합니다.

    Args:
        values: CLI/설정 파일에서 모은 인자 딕셔너리.

    Returns:
        ``report.json`` 내용.
    """
    started = time.perf_counter()
    metrics = select_metrics(values.get("metrics"), values.get("skip_metrics"), config.ALL_METRICS)
    has_linkage_args = bool(values.get("quasi_identifiers")) and bool(
        values.get("sensitive_attributes")
    )
    if config.LINKAGE_RISK in metrics and not has_linkage_args:
        logger.warning(
            "Skipping linkage_risk: --quasi_identifiers and --sensitive_attributes are required."
        )
        metrics.remove(config.LINKAGE_RISK)

    # 1. 임계값
    thresholds: dict[str, Any] = {}
    targets = [m for m in config.THRESHOLD_METRICS if m in metrics]
    if (
        values.get("reduction_method") == "inference_distance"
        and config.INFERENCE_RISK not in targets
        and values.get("inference_threshold") is None
    ):
        targets.append(config.INFERENCE_RISK)
    method = values.get("method") or config.DEFAULT_THRESHOLD_METHOD
    for metric in targets:
        m_method = (
            method
            if method in config.THRESHOLD_METHOD_SUPPORT[metric]
            else config.METHOD_SIMULATION
        )
        if m_method != method:
            logger.info("%s supports only %s; using it.", metric, m_method)
        opts = build_options(
            ThresholdOptions,
            values,
            metric=metric,
            method=m_method,
            strict=bool(values.get("strict")) and metric == config.SINGLE_OUT_RISK,
            manual_threshold=None,
        )
        thresholds[metric] = run_threshold(opts)

    # 2. 전처리
    pre = run_preprocess(build_options(PreprocessOptions, values))

    # 3. 지표
    m_opts = build_options(
        MetricOptions, values, synthetic=pre["processed_synthetic"], threshold_file=None
    )
    results = run_metrics(metrics, m_opts)

    summary = {}
    for name, r in results.items():
        entry = {
            "value": r.get("value"),
            "threshold": r.get("threshold"),
            "threshold_source": r.get("threshold_source"),
            "pass_rule": r.get("pass_rule"),
            "passed": r.get("passed"),
            "result_file": r.get("result_file"),
        }
        # linkage_risk 는 레코드 단위로 판정하므로 평균값과 임계값만으로는 판정을 알 수 없습니다.
        if r.get("exceed_count") is not None:
            entry.update(
                value_definition=r.get("value_definition"),
                exceed_count=r["exceed_count"],
                exceed_ratio=r["exceed_ratio"],
                n_evaluated_records=r["n_evaluated_records"],
            )
        summary[name] = entry
    judged = [s["passed"] for s in summary.values() if s["passed"] is not None]
    report = {
        "overall_passed": all(judged) if judged else None,
        "metrics": summary,
        "thresholds": {
            k: {
                "threshold": v["threshold"],
                "threshold_source": v["threshold_source"],
                "threshold_file": v["threshold_file"],
            }
            for k, v in thresholds.items()
        },
        "preprocess": {
            "validation_file": pre["validation_file"],
            "reduction_file": pre["reduction_file"],
            "processed_synthetic": pre["processed_synthetic"],
            "n_synthetic_before": pre["reduction"]["n_synthetic_before"],
            "n_synthetic_after": pre["reduction"]["n_synthetic_after"],
        },
        "parameters": {k: v for k, v in values.items() if k not in ("command", "func", "config")},
        "computed_at": io_utils.now_iso(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    path = io_utils.save_json(
        report, Path(values.get("output_dir") or config.DEFAULT_OUTPUT_DIR) / "report.json"
    )
    logger.info("Report saved to %s", path)
    for name, s in summary.items():
        verdict = "n/a" if s["passed"] is None else ("PASSED" if s["passed"] else "NOT PASSED")
        value = "n/a" if s["value"] is None else f"{s['value']:.6f}"
        logger.info("  %-22s value=%s  judgement=%s", name, value, verdict)
    return report
