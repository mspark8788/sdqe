"""임계값 계산 실행 (``sdqe threshold``).

임계값은 **원본 데이터만으로** 산출합니다. "원본 데이터가 가장 완벽한 재현 데이터"라는
가정 아래 귀무가설 하의 지표 분포를 만들고 그 분위수를 임계값으로 삼습니다.

분위수 의미: 0.95 를 쓰면 실제로 안전한 재현 데이터를 5% 확률로 안전하지 않다고
판정합니다. 기준을 강화하려면 0.90, 완화하려면 0.99 를 씁니다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sdqe import config, io_utils
from sdqe.metrics.utility.bivariate_similarity import association_matrix, constant_columns_of
from sdqe.metrics.utility.pmse import compute_pmse
from sdqe.preprocess.validation import prepare_original
from sdqe.threshold import distribution, simulation
from sdqe.threshold.store import threshold_path

logger = logging.getLogger("sdqe")


@dataclass
class ThresholdOptions:
    """임계값 계산 옵션. 필드 이름은 CLI 인자 이름과 같습니다."""

    metric: str
    original: str
    output_dir: str = config.DEFAULT_OUTPUT_DIR
    method: str = config.DEFAULT_THRESHOLD_METHOD
    n_simulations: int = config.DEFAULT_N_SIMULATIONS
    quantile: float = config.DEFAULT_QUANTILE
    random_seed: int = config.DEFAULT_RANDOM_SEED
    categorical_columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    size_correction: str = "on"
    duplicate_adjustment: str = "on"
    strict: bool = False
    manual_threshold: float | None = None
    distance_metric: str = config.DEFAULT_DISTANCE_METRIC
    use_gpu: bool = False
    n_jobs: int = config.DEFAULT_N_JOBS
    model: str = config.DEFAULT_PMSE_MODEL
    model_params: dict[str, Any] = field(default_factory=dict)
    drop_constant_columns: str = "on"


class ThresholdConfigError(ValueError):
    """지원하지 않는 임계값 계산 조합."""


def check_threshold_request(metric: str, method: str) -> None:
    """지표/방식 조합이 지원되는지 ``config.THRESHOLD_METHOD_SUPPORT`` 로 확인합니다.

    Raises:
        ThresholdConfigError: 계산 대상이 아니거나 지원하지 않는 방식일 때.
    """
    if metric == config.LINKAGE_RISK:
        raise ThresholdConfigError(
            "linkage_risk has no computed threshold; set it with --linkage_threshold "
            f"(default {config.DEFAULT_LINKAGE_THRESHOLD}, must be < 1.0)."
        )
    if metric not in config.THRESHOLD_METHOD_SUPPORT:
        raise ThresholdConfigError(
            f"'{metric}' is not a threshold target. Supported metrics: "
            f"{', '.join(config.THRESHOLD_METRICS)}"
        )
    supported = config.THRESHOLD_METHOD_SUPPORT[metric]
    if method not in supported:
        raise ThresholdConfigError(
            f"--method {method} is not supported for metric '{metric}'.\n"
            f"       Supported methods: {', '.join(supported)}"
        )


def _parameters(opts: ThresholdOptions) -> dict[str, Any]:
    params: dict[str, Any] = {
        "method": opts.method,
        "quantile": opts.quantile,
        "random_seed": opts.random_seed,
    }
    if opts.method == config.METHOD_SIMULATION:
        params["n_simulations"] = opts.n_simulations
    if opts.metric == config.SINGLE_OUT_RISK:
        params["duplicate_adjustment"] = opts.duplicate_adjustment
        params["size_correction"] = (
            opts.size_correction if opts.method == config.METHOD_SIMULATION else "not_applicable"
        )
    if opts.metric == config.INFERENCE_RISK:
        params["distance_metric"] = opts.distance_metric
        params["use_gpu"] = opts.use_gpu
    if opts.metric == config.PMSE:
        params["model"] = opts.model
        params["model_params"] = opts.model_params or {}
    if opts.metric == config.BIVARIATE_SIMILARITY:
        params["drop_constant_columns"] = opts.drop_constant_columns
    return params


def _summary_stats(values: np.ndarray, quantile: float) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    return {
        "n_valid": int(len(finite)),
        "n_invalid": int(len(values) - len(finite)),
        "mean": float(finite.mean()) if len(finite) else None,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "quantile_value": float(np.quantile(finite, quantile)) if len(finite) else None,
    }


def run_threshold(opts: ThresholdOptions) -> dict[str, Any]:
    """임계값을 계산하고 ``output_dir/threshold/<metric>.json`` 으로 저장합니다.

    Args:
        opts: 임계값 계산 옵션.

    Returns:
        저장한 요약 JSON 내용.

    References:
        [2] 박민수 외 (2024), 3장.
    """
    started = time.perf_counter()
    metric = opts.metric
    if opts.strict and metric != config.SINGLE_OUT_RISK:
        raise ThresholdConfigError("--strict is only available for single_out_risk.")
    user_defined = opts.strict or opts.manual_threshold is not None
    if not user_defined:
        check_threshold_request(metric, opts.method)
    elif metric not in config.THRESHOLD_METHOD_SUPPORT:
        check_threshold_request(metric, config.METHOD_SIMULATION)
    if not 0.0 < opts.quantile < 1.0:
        raise ThresholdConfigError(f"--quantile must be between 0 and 1; got {opts.quantile}.")

    out_dir = io_utils.stage_dir(opts.output_dir, "threshold")
    summary: dict[str, Any] = {"metric": metric}

    if user_defined:
        value = 0.0 if opts.strict else float(opts.manual_threshold)
        logger.info("%s threshold set by user: %s (no computation)", metric, value)
        summary.update(
            {
                "threshold": value,
                "threshold_source": config.THRESHOLD_SOURCE_USER,
                "method": "strict" if opts.strict else "manual",
                "parameters": {"strict": opts.strict, "manual_threshold": opts.manual_threshold},
            }
        )
        return _finish(summary, opts, out_dir, started)

    original_raw = io_utils.read_table(opts.original)
    original, schema = prepare_original(
        original_raw, opts.categorical_columns, opts.exclude_columns
    )
    n = len(original)
    summary["input_info"] = {"n_original": n, "n_columns": len(schema.columns)}
    summary["excluded_columns"] = [
        c for c in (opts.exclude_columns or []) if c in original_raw.columns
    ]
    summary["method"] = opts.method
    summary["parameters"] = _parameters(opts)
    logger.info(
        "Computing %s threshold (%s, quantile %.3f): a quantile of %.2f means safe synthetic "
        "data is judged unsafe with probability %.0f%%.",
        metric,
        opts.method,
        opts.quantile,
        opts.quantile,
        (1 - opts.quantile) * 100,
    )
    if (
        opts.method == config.METHOD_SIMULATION
        and n > config.LARGE_DATA_ROWS
        and metric in (config.SINGLE_OUT_RISK, config.INFERENCE_RISK)
    ):
        logger.info(
            "Large data (%d rows): --method distribution gives the %s threshold with a single "
            "computation and is much faster.",
            n,
            metric,
        )

    if opts.method == config.METHOD_DISTRIBUTION:
        if metric == config.SINGLE_OUT_RISK:
            p = distribution.single_out_risk_proportion(
                original, schema, opts.duplicate_adjustment == "on"
            )
            extra: dict[str, Any] = {}
            n_used = n
        else:
            p, extra = distribution.inference_risk_proportion(
                original, schema, opts.distance_metric, opts.use_gpu, opts.n_jobs
            )
            n_used = extra["n_evaluated"]  # 동률을 뺀 레코드 수
        dist = distribution.normal_quantile_threshold(p, n_used, opts.quantile)
        summary.update(
            {
                "threshold": dist.pop("threshold"),
                "threshold_source": config.THRESHOLD_SOURCE_DISTRIBUTION,
                "distribution": {
                    "family": "normal approximation of binomial proportion",
                    **dist,
                    **extra,
                },
            }
        )
        return _finish(summary, opts, out_dir, started)

    # ---- 시뮬레이션 ----
    if opts.n_simulations < config.MIN_RECOMMENDED_SIMULATIONS:
        logger.warning(
            "--n_simulations %d is below %d; reference [2] recommends at least %d repetitions "
            "for a stable threshold.",
            opts.n_simulations,
            config.MIN_RECOMMENDED_SIMULATIONS,
            config.MIN_RECOMMENDED_SIMULATIONS,
        )
    seeds = simulation.derive_seeds(opts.random_seed, opts.n_simulations)
    raw: dict[str, Any] = {"metric": metric, "seeds": seeds}
    info: dict[str, Any] = {}
    if metric == config.SINGLE_OUT_RISK:
        values = simulation.simulate_single_out_risk(
            original, schema, seeds, opts.duplicate_adjustment == "on"
        )
    elif metric == config.INFERENCE_RISK:
        values, info = simulation.simulate_inference_risk(
            original, schema, seeds, opts.distance_metric, opts.use_gpu, opts.n_jobs
        )
    elif metric == config.BIVARIATE_SIMILARITY:
        cols_schema = schema
        excluded = []
        if opts.drop_constant_columns == "on":
            excluded = constant_columns_of(original, original, schema)
            cols_schema = schema.subset([c for c in schema.columns if c not in excluded])
        info["constant_columns_excluded"] = excluded

        def bivariate_fn(a, b):
            with np.errstate(all="ignore"):
                ma, _ = association_matrix(a, cols_schema.columns, cols_schema)
                mb, _ = association_matrix(b, cols_schema.columns, cols_schema)
            return float(np.std(ma - mb))

        values = simulation.simulate_generic(original, seeds, bivariate_fn, metric)
    else:
        values = simulation.simulate_generic(
            original,
            seeds,
            lambda a, b: compute_pmse(
                a, b, schema, opts.model, opts.model_params, opts.random_seed
            )["value"],
            metric,
        )
    raw["values"] = values
    stats = _summary_stats(values, opts.quantile)
    if stats["n_invalid"]:
        logger.warning("%d simulation(s) produced NaN and were ignored.", stats["n_invalid"])
    if stats["n_valid"] == 0:
        raise RuntimeError(f"All {metric} simulations produced NaN; no threshold available.")
    threshold = stats["quantile_value"]
    if metric == config.SINGLE_OUT_RISK:
        p_hat = threshold
        p_star = float(simulation.size_correction(p_hat))
        raw["p_star_values"] = simulation.size_correction(values)
        summary["single_out_risk"] = {
            "p_hat": p_hat,
            "p_star": p_star,
            "size_correction": opts.size_correction,
            "size_correction_formula": "p_star = 1 - (1 - p_hat)^2 (equation 5)",
        }
        if opts.size_correction == "on":
            threshold = p_star
        else:
            logger.warning(
                "--size_correction off: the threshold is for half-size data. Compare it with the "
                "metric computed on a random half of the original data."
            )
    if metric in (config.BIVARIATE_SIMILARITY, config.PMSE):
        summary["note"] = config.EXTENDED_SCOPE_NOTE
    sim_path = io_utils.save_json(raw, out_dir / f"{metric}_simulations.json")
    summary.update(
        {
            "threshold": threshold,
            "threshold_source": config.THRESHOLD_SOURCE_SIMULATION,
            "simulation": {
                "n_simulations": opts.n_simulations,
                **stats,
                **info,
                "simulations_file": sim_path.as_posix(),
            },
        }
    )
    return _finish(summary, opts, out_dir, started)


def _finish(
    summary: dict[str, Any], opts: ThresholdOptions, out_dir: Path, started: float
) -> dict[str, Any]:
    summary["options"] = {k: v for k, v in asdict(opts).items() if k not in ("output_dir",)}
    summary["computed_at"] = io_utils.now_iso()
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    path = io_utils.save_json(summary, threshold_path(opts.output_dir, opts.metric))
    logger.info(
        "Threshold %s = %.6f (source: %s). Saved to %s",
        opts.metric,
        summary["threshold"],
        summary["threshold_source"],
        path,
    )
    summary["threshold_file"] = path.as_posix()
    return summary
