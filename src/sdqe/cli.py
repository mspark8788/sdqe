"""``sdqe`` 명령행 진입점.

서브커맨드
- ``preprocess``: 정합성 검증 + 초과 레코드 처리
- ``threshold`` : 지표 1개의 임계값 계산 (원본만 사용)
- ``metric``    : 지표 1개 계산
- ``utility``   : 유용성 지표 일괄 계산
- ``privacy``   : 안전성 지표 일괄 계산
- ``report``    : 전처리부터 전 지표까지 실행 후 통합 리포트

그룹/전체 모드(``utility``, ``privacy``, ``report``)는 ``--config config.yaml`` 로
설정을 받을 수 있고,
CLI 인자가 설정 파일보다 우선합니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from sdqe import __version__, config
from sdqe.metrics.runner import MetricOptions, run_metrics
from sdqe.preprocess.pipeline import PreprocessOptions, run_preprocess
from sdqe.preprocess.validation import DataValidationError
from sdqe.report import build_options, run_report, select_metrics
from sdqe.threshold.runner import ThresholdConfigError, ThresholdOptions, run_threshold

logger = logging.getLogger("sdqe")

CONFIG_COMMANDS = ("utility", "privacy", "report")


class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def setup_logging(verbose: bool) -> None:
    """stdout 핸들러 하나만 둔 로거를 설정합니다."""
    root = logging.getLogger("sdqe")
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False


def json_dict(text: str) -> dict[str, Any]:
    """``--model_params`` JSON 문자열을 딕셔너리로 바꿉니다."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object, e.g. '{\"max_depth\": 5}'")
    return value


def _quantile(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 (exclusive)")
    return value


# ---------------------------------------------------------------------------
# 인자 그룹
# ---------------------------------------------------------------------------
def add_data_args(
    p: argparse.ArgumentParser, synthetic: bool = True, required: bool = True
) -> None:
    g = p.add_argument_group("data")
    g.add_argument(
        "--original", required=required, help="Path to the original data (csv, parquet, xlsx)."
    )
    if synthetic:
        g.add_argument("--synthetic", required=required, help="Path to the synthetic data.")
    g.add_argument("--output_dir", default=config.DEFAULT_OUTPUT_DIR, help="Directory for results.")
    g.add_argument(
        "--categorical_columns",
        nargs="+",
        metavar="COL",
        default=None,
        help="Columns treated as categorical. If omitted, inferred from dtypes "
        "(non-numeric columns are categorical).",
    )
    g.add_argument(
        "--exclude_columns",
        nargs="+",
        metavar="COL",
        default=None,
        help="Columns excluded from all computations.",
    )
    g.add_argument(
        "--random_seed",
        type=int,
        default=config.DEFAULT_RANDOM_SEED,
        help="Seed for every random step (splits, sampling, models).",
    )
    g.add_argument(
        "--n_jobs",
        type=int,
        default=config.DEFAULT_N_JOBS,
        help="Number of threads for distance computation (-1 = all cores).",
    )
    g.add_argument("--verbose", action="store_true", help="Show debug logs.")


def add_distance_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("distance (inference_risk)")
    g.add_argument(
        "--distance_metric",
        choices=config.DISTANCE_METRICS,
        default=config.DEFAULT_DISTANCE_METRIC,
        help="Distance for nearest_distance/reference_distance. gower: category mismatch + "
        "range-scaled absolute difference; cosine: category match + cosine of "
        "standardized numeric values.",
    )
    g.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use a CUDA GPU (torch) for cosine distance; falls back to CPU if unavailable.",
    )


def add_single_out_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--duplicate_adjustment",
        choices=config.ON_OFF,
        default="on",
        help="single_out_risk formula. on: equation (2), divides each match by the number "
        "of identical original records; off: equation (1), simple match ratio.",
    )


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("pmse model")
    g.add_argument(
        "--model",
        choices=config.PMSE_MODELS,
        default=config.DEFAULT_PMSE_MODEL,
        help="Classifier used for pmse propensity scores.",
    )
    g.add_argument(
        "--model_params",
        type=json_dict,
        default={},
        help="Model parameters as a JSON object, e.g. '{\"max_depth\": 5}'.",
    )


def add_bivariate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--drop_constant_columns",
        choices=config.ON_OFF,
        default="on",
        help="Exclude constant columns (all values identical) from bivariate_similarity; "
        "excluded columns are recorded in the result.",
    )


def add_linkage_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("linkage_risk (CAP)")
    g.add_argument(
        "--quasi_identifiers",
        nargs="+",
        metavar="COL",
        default=None,
        help="Quasi-identifier columns (required for linkage_risk).",
    )
    g.add_argument(
        "--sensitive_attributes",
        nargs="+",
        metavar="COL",
        default=None,
        help="Sensitive attribute columns (required for linkage_risk).",
    )
    g.add_argument(
        "--linkage_threshold",
        type=float,
        default=config.DEFAULT_LINKAGE_THRESHOLD,
        help="User-defined CAP threshold, 0.0 <= value < 1.0. Records with CAP > threshold "
        "are reported; values below 0.5 trigger a utility warning.",
    )
    g.add_argument(
        "--linkage_missing_policy",
        choices=config.LINKAGE_MISSING_POLICIES,
        default=config.DEFAULT_LINKAGE_MISSING_POLICY,
        help="exclude: skip rows with missing key/sensitive values; error: stop.",
    )
    g.add_argument(
        "--linkage_evaluation_mode",
        choices=config.LINKAGE_EVALUATION_MODES,
        default=config.DEFAULT_LINKAGE_EVALUATION_MODE,
        help="individual: each sensitive attribute; combined: also all sensitive "
        "attributes jointly.",
    )


def add_threshold_args(p: argparse.ArgumentParser, with_metric: bool) -> None:
    g = p.add_argument_group("threshold")
    if with_metric:
        g.add_argument(
            "--metric",
            required=True,
            metavar="{" + ",".join(config.THRESHOLD_METRICS) + "}",
            help="Metric whose threshold is computed. linkage_risk is user-defined "
            "(--linkage_threshold) and univariate_similarity has no threshold.",
        )
    g.add_argument(
        "--method",
        choices=config.THRESHOLD_METHODS,
        default=config.DEFAULT_THRESHOLD_METHOD,
        help="simulation: repeated 50:50 splits of the original data (all 4 metrics); "
        "distribution: normal approximation N(p, p(1-p)/n), single computation "
        "(single_out_risk, inference_risk only).",
    )
    g.add_argument(
        "--n_simulations",
        type=int,
        default=config.DEFAULT_N_SIMULATIONS,
        help="Number of simulation repetitions (100 or more recommended).",
    )
    g.add_argument(
        "--quantile",
        type=_quantile,
        default=config.DEFAULT_QUANTILE,
        help="Quantile used as the threshold. 0.95 means safe synthetic data is judged "
        "unsafe with 5%% probability; use 0.90 to tighten, 0.99 to relax.",
    )
    g.add_argument(
        "--size_correction",
        choices=config.ON_OFF,
        default="on",
        help="single_out_risk simulation only: apply equation (5) p* = 1-(1-p)^2 to "
        "correct the n/2 split size.",
    )
    g.add_argument(
        "--strict",
        action="store_true",
        help="single_out_risk only: set the threshold to 0 without computation.",
    )
    if with_metric:
        g.add_argument(
            "--manual_threshold",
            type=float,
            default=None,
            help="Use this threshold without computation (recorded as user_defined).",
        )


def add_preprocess_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("preprocess")
    g.add_argument(
        "--reduction_method",
        choices=config.REDUCTION_METHODS,
        default=None,
        help="How to delete excess synthetic records when synthetic > original. "
        "random; pmse_probability: delete records most likely judged synthetic; "
        "inference_distance: delete by inference_risk indicator to meet its threshold. "
        "If omitted, no records are deleted.",
    )
    g.add_argument(
        "--remove_exact_matches",
        choices=config.ON_OFF,
        default="off",
        help="Remove synthetic records identical to an original record before reduction.",
    )
    g.add_argument(
        "--inference_threshold",
        type=float,
        default=None,
        help="inference_distance: threshold to use instead of "
        "<output_dir>/threshold/inference_risk.json.",
    )
    g.add_argument(
        "--target_margin",
        type=float,
        default=config.DEFAULT_TARGET_MARGIN,
        help="inference_distance: target = threshold * target_margin.",
    )


def add_group_args(p: argparse.ArgumentParser, universe: Sequence[str]) -> None:
    p.add_argument(
        "--config", default=None, help="YAML file with argument values (CLI arguments win)."
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=universe,
        default=None,
        help="Metrics to compute (default: all in the group).",
    )
    p.add_argument(
        "--skip_metrics",
        nargs="+",
        choices=universe,
        default=None,
        help="Metrics to skip, e.g. bivariate_similarity.",
    )


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    """argparse 파서와 서브커맨드별 파서를 만듭니다."""
    parser = argparse.ArgumentParser(
        prog="sdqe",
        formatter_class=_Formatter,
        description="Synthetic Data Quality Evaluation: utility and privacy metrics.",
    )
    parser.add_argument("--version", action="version", version=f"sdqe {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)
    parsers: dict[str, argparse.ArgumentParser] = {}

    p = subs.add_parser(
        "preprocess",
        formatter_class=_Formatter,
        help="Validate original/synthetic consistency and reduce excess records.",
    )
    add_data_args(p)
    add_preprocess_args(p)
    add_distance_args(p)
    add_model_args(p)
    p.add_argument(
        "--threshold_file",
        default=None,
        help="inference_risk threshold JSON (default: <output_dir>/threshold/inference_risk.json).",
    )
    parsers["preprocess"] = p

    p = subs.add_parser(
        "threshold",
        formatter_class=_Formatter,
        help="Compute the threshold of one metric from the original data only.",
    )
    add_data_args(p, synthetic=False)
    add_threshold_args(p, with_metric=True)
    add_single_out_args(p)
    add_distance_args(p)
    add_model_args(p)
    add_bivariate_args(p)
    parsers["threshold"] = p

    p = subs.add_parser("metric", formatter_class=_Formatter, help="Compute one metric.")
    add_data_args(p)
    p.add_argument("--name", required=True, choices=config.ALL_METRICS, help="Metric to compute.")
    p.add_argument(
        "--threshold_file",
        default=None,
        help="Threshold JSON for judgement (default: <output_dir>/threshold/<name>.json).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=config.DEFAULT_ALPHA,
        help="univariate_similarity: significance level for the summary count.",
    )
    p.add_argument(
        "--plots", choices=config.ON_OFF, default="on", help="univariate_similarity: save plots."
    )
    add_single_out_args(p)
    add_distance_args(p)
    add_model_args(p)
    add_bivariate_args(p)
    add_linkage_args(p)
    parsers["metric"] = p

    p = subs.add_parser("utility", formatter_class=_Formatter, help="Compute all utility metrics.")
    add_data_args(p, required=False)
    add_group_args(p, config.UTILITY_METRICS)
    p.add_argument(
        "--alpha",
        type=float,
        default=config.DEFAULT_ALPHA,
        help="univariate_similarity: significance level for the summary count.",
    )
    p.add_argument(
        "--plots", choices=config.ON_OFF, default="on", help="univariate_similarity: save plots."
    )
    add_model_args(p)
    add_bivariate_args(p)
    parsers["utility"] = p

    p = subs.add_parser("privacy", formatter_class=_Formatter, help="Compute all privacy metrics.")
    add_data_args(p, required=False)
    add_group_args(p, config.PRIVACY_METRICS)
    add_single_out_args(p)
    add_distance_args(p)
    add_linkage_args(p)
    parsers["privacy"] = p

    p = subs.add_parser(
        "report",
        formatter_class=_Formatter,
        help="Run thresholds, preprocessing and all metrics, then write report.json.",
    )
    add_data_args(p, required=False)
    add_group_args(p, config.ALL_METRICS)
    add_preprocess_args(p)
    add_threshold_args(p, with_metric=False)
    add_single_out_args(p)
    add_distance_args(p)
    add_model_args(p)
    add_bivariate_args(p)
    add_linkage_args(p)
    p.add_argument(
        "--alpha",
        type=float,
        default=config.DEFAULT_ALPHA,
        help="univariate_similarity: significance level for the summary count.",
    )
    p.add_argument(
        "--plots", choices=config.ON_OFF, default="on", help="univariate_similarity: save plots."
    )
    parsers["report"] = p
    return parser, parsers


def load_config_file(path: str, subparser: argparse.ArgumentParser) -> dict[str, Any]:
    """YAML 설정을 읽고 알 수 없는 키가 있으면 에러를 냅니다."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping of argument names to values.")
    known = {a.dest for a in subparser._actions}  # noqa: SLF001
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown keys in config file {path}: {unknown}")
    if isinstance(data.get("model_params"), str):
        data["model_params"] = json_dict(data["model_params"])
    return data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """설정 파일 기본값을 반영해 인자를 해석합니다 (CLI 인자가 우선)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser, parsers = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    command = next((a for a in argv if not a.startswith("-")), None)
    if known.config:
        if command not in CONFIG_COMMANDS:
            parser.error(f"--config is available for {', '.join(CONFIG_COMMANDS)} only.")
        parsers[command].set_defaults(**load_config_file(known.config, parsers[command]))
    args = parser.parse_args(argv)
    if args.command in CONFIG_COMMANDS:
        missing = [n for n in ("original", "synthetic") if not getattr(args, n, None)]
        if missing:
            parsers[args.command].error(
                "the following arguments are required (CLI or --config): "
                + ", ".join(f"--{m}" for m in missing)
            )
    return args


def _metric_options(values: dict[str, Any], **overrides: Any) -> MetricOptions:
    values = dict(values)
    values["make_plots"] = values.get("plots", "on") == "on"
    return build_options(MetricOptions, values, **overrides)


def dispatch(args: argparse.Namespace) -> Any:
    """서브커맨드를 실행합니다."""
    values = vars(args)
    if args.command == "preprocess":
        return run_preprocess(build_options(PreprocessOptions, values))
    if args.command == "threshold":
        return run_threshold(build_options(ThresholdOptions, values))
    if args.command == "metric":
        return run_metrics([args.name], _metric_options(values))
    if args.command in ("utility", "privacy"):
        universe = config.UTILITY_METRICS if args.command == "utility" else config.PRIVACY_METRICS
        names = select_metrics(args.metrics, args.skip_metrics, universe)
        if (
            args.command == "privacy"
            and config.LINKAGE_RISK in names
            and not (args.quasi_identifiers and args.sensitive_attributes)
        ):
            logger.warning(
                "Skipping linkage_risk: --quasi_identifiers and --sensitive_attributes "
                "are required."
            )
            names.remove(config.LINKAGE_RISK)
        return run_metrics(names, _metric_options(values, threshold_file=None))
    if args.command == "report":
        values["make_plots"] = values.get("plots", "on") == "on"
        return run_report(values)
    raise ValueError(f"Unknown command {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 진입점. 성공 시 0, 입력 오류 시 2, 그 밖의 오류 시 1 을 반환합니다."""
    args = parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    try:
        dispatch(args)
    except (DataValidationError, ThresholdConfigError, ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        if getattr(args, "verbose", False):
            logger.exception("Details")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
