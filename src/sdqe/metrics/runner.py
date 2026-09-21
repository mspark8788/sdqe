"""지표 계산 실행 (``sdqe metric``, ``sdqe utility``, ``sdqe privacy``).

데이터는 한 번만 읽고 검증한 뒤 여러 지표에 재사용합니다. 결과는
``output_dir/metrics/<metric>.json`` 으로 저장하며 명세 7장의 공통 필드를 채웁니다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sdqe import config, io_utils
from sdqe.metrics.privacy.inference_risk import (
    load_or_compute_inference_details,
    summarize_inference,
)
from sdqe.metrics.privacy.linkage_risk import CSV_NA_VALUES, compute_linkage_risk
from sdqe.metrics.privacy.single_out_risk import compute_single_out_risk
from sdqe.metrics.utility.bivariate_similarity import compute_bivariate_similarity
from sdqe.metrics.utility.pmse import compute_pmse
from sdqe.metrics.utility.univariate_similarity import compute_univariate_similarity
from sdqe.preprocess.validation import ValidationResult, validate_pair
from sdqe.threshold.store import judge, load_threshold_record

logger = logging.getLogger("sdqe")


@dataclass
class MetricOptions:
    """지표 계산 옵션. 필드 이름은 CLI 인자 이름과 같습니다."""

    original: str
    synthetic: str
    output_dir: str = config.DEFAULT_OUTPUT_DIR
    categorical_columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    random_seed: int = config.DEFAULT_RANDOM_SEED
    n_jobs: int = config.DEFAULT_N_JOBS
    distance_metric: str = config.DEFAULT_DISTANCE_METRIC
    use_gpu: bool = False
    duplicate_adjustment: str = "on"
    model: str = config.DEFAULT_PMSE_MODEL
    model_params: dict[str, Any] = field(default_factory=dict)
    drop_constant_columns: str = "on"
    quasi_identifiers: list[str] | None = None
    sensitive_attributes: list[str] | None = None
    linkage_threshold: float = config.DEFAULT_LINKAGE_THRESHOLD
    linkage_missing_policy: str = config.DEFAULT_LINKAGE_MISSING_POLICY
    linkage_evaluation_mode: str = config.DEFAULT_LINKAGE_EVALUATION_MODE
    alpha: float = config.DEFAULT_ALPHA
    threshold_file: str | None = None
    make_plots: bool = True


def _input_info(data: ValidationResult) -> dict[str, int]:
    return {
        "n_original": len(data.original),
        "n_synthetic": len(data.synthetic),
        "n_columns": len(data.schema.columns),
    }


def _metric_parameters(name: str, opts: MetricOptions) -> dict[str, Any]:
    params: dict[str, Any] = {"random_seed": opts.random_seed}
    if name == config.SINGLE_OUT_RISK:
        params["duplicate_adjustment"] = opts.duplicate_adjustment
    elif name == config.INFERENCE_RISK:
        params.update(distance_metric=opts.distance_metric, use_gpu=opts.use_gpu)
    elif name == config.PMSE:
        params.update(model=opts.model, model_params=opts.model_params or {})
    elif name == config.BIVARIATE_SIMILARITY:
        params["drop_constant_columns"] = opts.drop_constant_columns
    elif name == config.UNIVARIATE_SIMILARITY:
        params["alpha"] = opts.alpha
    elif name == config.LINKAGE_RISK:
        params.update(
            quasi_identifiers=opts.quasi_identifiers,
            sensitive_attributes=opts.sensitive_attributes,
            linkage_threshold=opts.linkage_threshold,
            linkage_missing_policy=opts.linkage_missing_policy,
            linkage_evaluation_mode=opts.linkage_evaluation_mode,
        )
    return params


def compute_metric(name: str, data: ValidationResult, opts: MetricOptions) -> dict[str, Any]:
    """지표 하나를 계산하고 판정까지 마친 결과 딕셔너리를 반환합니다 (저장은 하지 않음)."""
    started = time.perf_counter()
    params = _metric_parameters(name, opts)
    result: dict[str, Any] = {"metric": name, "value": None}
    excluded = list(data.report.get("excluded_columns", []))
    extra: dict[str, Any] = {}
    input_info = _input_info(data)

    if name == config.UNIVARIATE_SIMILARITY:
        figures = io_utils.stage_dir(opts.output_dir, "figures") if opts.make_plots else None
        details = compute_univariate_similarity(
            data.original, data.synthetic, data.schema, figures, opts.alpha
        )
        result["value"] = details.pop("value")
        extra["details"] = details
    elif name == config.BIVARIATE_SIMILARITY:
        details = compute_bivariate_similarity(
            data.original, data.synthetic, data.schema, opts.drop_constant_columns == "on"
        )
        result["value"] = details.pop("value")
        excluded += details.pop("excluded_columns")
        extra["details"] = details
    elif name == config.PMSE:
        details = compute_pmse(
            data.original,
            data.synthetic,
            data.schema,
            opts.model,
            opts.model_params,
            opts.random_seed,
        )
        result["value"] = details.pop("value")
        extra["details"] = details
    elif name == config.SINGLE_OUT_RISK:
        details = compute_single_out_risk(
            data.original, data.synthetic, data.schema, opts.duplicate_adjustment == "on"
        )
        result["value"] = details.pop("value")
        extra["details"] = details
    elif name == config.INFERENCE_RISK:
        inf, cache_hit, key = load_or_compute_inference_details(
            data.original,
            data.synthetic,
            data.schema,
            opts.distance_metric,
            opts.original,
            opts.synthetic,
            opts.output_dir,
            opts.use_gpu,
            opts.n_jobs,
        )
        summary = summarize_inference(inf)
        result["value"] = summary.pop("value")
        result["reference_value"] = summary.pop("reference_value")
        extra["cache_hit"] = cache_hit
        extra["cache_key"] = key
        extra["details"] = summary
        logger.info(
            "inference_risk %.6f (reference %.1f): %s",
            result["value"],
            result["reference_value"],
            summary["interpretation"],
        )
    elif name == config.LINKAGE_RISK:
        if not opts.quasi_identifiers or not opts.sensitive_attributes:
            raise ValueError(
                "linkage_risk requires --quasi_identifiers and --sensitive_attributes."
            )
        # 기존 구현과 같이 값을 문자열 그대로 비교합니다.
        orig_s = io_utils.read_table_as_strings(opts.original, CSV_NA_VALUES)
        syn_s = io_utils.read_table_as_strings(opts.synthetic, CSV_NA_VALUES)
        summary, records = compute_linkage_risk(
            orig_s,
            syn_s,
            opts.quasi_identifiers,
            opts.sensitive_attributes,
            opts.linkage_threshold,
            opts.linkage_missing_policy,
            opts.linkage_evaluation_mode,
        )
        records_path = io_utils.save_json(
            {"metric": name, "records": records.to_dict(orient="records")},
            Path(opts.output_dir) / "metrics" / f"{name}_records.json",
        )
        result["value"] = summary.pop("value")
        result["value_definition"] = summary.pop("value_definition")
        result["threshold"] = summary.pop("threshold")
        result["threshold_source"] = summary.pop("threshold_source")
        result["pass_rule"] = summary.pop("pass_rule")
        result["passed"] = summary.pop("passed")
        result["exceed_count"] = summary.pop("exceed_count")
        result["exceed_ratio"] = summary.pop("exceed_ratio")
        result["n_evaluated_records"] = summary.pop("n_evaluated_records")
        input_info = {
            "n_original": len(orig_s),
            "n_synthetic": len(syn_s),
            "n_columns": len(orig_s.columns),
        }
        extra["details"] = {**summary, "records_file": records_path.as_posix()}
    else:
        raise ValueError(f"Unknown metric '{name}'. Use one of {config.ALL_METRICS}.")

    if name in config.PASS_RULES:
        record, path = load_threshold_record(
            opts.output_dir, name, opts.threshold_file if opts.threshold_file else None
        )
        verdict = judge(name, result["value"], record, path, params)
        warnings = verdict.pop("judge_warnings")
        result.update(verdict)
        if warnings:
            extra["judge_warnings"] = warnings
    result.update(
        {
            "parameters": params,
            "input_info": input_info,
            "excluded_columns": excluded,
        }
    )
    if name == config.INFERENCE_RISK:
        result["cache_hit"] = extra.pop("cache_hit")
    result.update(extra)
    result["computed_at"] = io_utils.now_iso()
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def log_result(result: dict[str, Any]) -> None:
    """지표 결과를 평문 한 줄로 출력합니다."""
    value = result.get("value")
    text = "n/a" if value is None else f"{value:.6f}"
    thr = result.get("threshold")
    verdict = (
        "" if result.get("passed") is None else ("PASSED" if result["passed"] else "NOT PASSED")
    )
    thr_text = "" if thr is None else f" | threshold {thr:.6f} ({result.get('threshold_source')})"
    if result.get("exceed_count") is not None:
        thr_text += (
            f" | {result['exceed_count']} of {result['n_evaluated_records']} records "
            f"({result['exceed_ratio']:.1%}) exceed it"
        )
    logger.info(
        "%s (%s) = %s%s %s",
        result["metric"],
        config.METRIC_KOREAN_NAMES[result["metric"]],
        text,
        thr_text,
        verdict,
    )


def run_metrics(names: Sequence[str], opts: MetricOptions) -> dict[str, dict[str, Any]]:
    """여러 지표를 계산해 각각 JSON 으로 저장합니다.

    Args:
        names: 지표 식별자 목록.
        opts: 지표 계산 옵션.

    Returns:
        지표별 결과 딕셔너리.
    """
    unknown = [n for n in names if n not in config.ALL_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}. Use {config.ALL_METRICS}.")
    if opts.threshold_file and len(names) > 1:
        raise ValueError("--threshold_file can be used with a single metric only.")
    data = validate_pair(
        io_utils.read_table(opts.original),
        io_utils.read_table(opts.synthetic),
        opts.categorical_columns,
        opts.exclude_columns,
    )
    out_dir = io_utils.stage_dir(opts.output_dir, "metrics")
    results = {}
    for name in names:
        logger.info("Computing %s ...", name)
        result = compute_metric(name, data, opts)
        path = io_utils.save_json(result, out_dir / f"{name}.json")
        result["result_file"] = path.as_posix()
        log_result(result)
        results[name] = result
    return results
