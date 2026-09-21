"""전처리 실행 (``sdqe preprocess``, 명세 3장).

1. 정합성 검증 -> ``preprocess/validation.json``
2. (선택) 원본과 완전히 같은 합성 레코드 제거 (기존 Kdata 파이프라인의 동작, 기본 off)
3. (선택) 초과 레코드 삭제 -> ``preprocess/reduction.json``
4. 처리된 합성 데이터 저장 -> ``preprocess/synthetic_processed.csv``
   값은 원자료 표기를 그대로 보존하고 열 순서만 원본에 맞춥니다.
5. ``inference_distance`` 전략을 썼다면 처리된 파일 기준의 추론위험도 캐시도 저장합니다.
   이후 ``sdqe metric --name inference_risk --synthetic <처리된 파일>`` 은 캐시를 재사용합니다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sdqe import config, io_utils
from sdqe.metrics.privacy.inference_risk import inference_cache_key, store_inference_cache
from sdqe.metrics.privacy.single_out_risk import record_group_ids
from sdqe.preprocess.reduction import ReductionContext, output_path_for, reduce_records
from sdqe.preprocess.validation import validate_pair
from sdqe.threshold.store import check_distance_metric, load_threshold_record

logger = logging.getLogger("sdqe")


@dataclass
class PreprocessOptions:
    """전처리 옵션. 필드 이름은 CLI 인자 이름과 같습니다."""

    original: str
    synthetic: str
    output_dir: str = config.DEFAULT_OUTPUT_DIR
    categorical_columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    reduction_method: str | None = None
    remove_exact_matches: str = "off"
    random_seed: int = config.DEFAULT_RANDOM_SEED
    n_jobs: int = config.DEFAULT_N_JOBS
    distance_metric: str = config.DEFAULT_DISTANCE_METRIC
    use_gpu: bool = False
    inference_threshold: float | None = None
    threshold_file: str | None = None
    target_margin: float = config.DEFAULT_TARGET_MARGIN
    model: str = config.DEFAULT_PMSE_MODEL
    model_params: dict[str, Any] = field(default_factory=dict)


def _resolve_inference_threshold(opts: PreprocessOptions) -> tuple[float | None, str | None]:
    if opts.inference_threshold is not None:
        return float(opts.inference_threshold), "--inference_threshold"
    record, path = load_threshold_record(
        opts.output_dir, config.INFERENCE_RISK, opts.threshold_file
    )
    if record is None:
        return None, None
    check_distance_metric(record, opts.distance_metric, path)
    return float(record["threshold"]), path.as_posix()


def run_preprocess(opts: PreprocessOptions) -> dict[str, Any]:
    """전처리를 실행하고 결과 파일을 저장합니다.

    Args:
        opts: 전처리 옵션.

    Returns:
        ``validation``, ``reduction``, ``processed_synthetic`` 를 담은 요약.
    """
    started = time.perf_counter()
    out_dir = io_utils.stage_dir(opts.output_dir, "preprocess")
    original_raw = io_utils.read_table(opts.original)
    synthetic_raw = io_utils.read_table(opts.synthetic)
    data = validate_pair(
        original_raw, synthetic_raw, opts.categorical_columns, opts.exclude_columns
    )
    common = {
        "parameters": {k: v for k, v in asdict(opts).items() if k != "output_dir"},
        "computed_at": io_utils.now_iso(),
    }
    validation_path = io_utils.save_json({**data.report, **common}, out_dir / "validation.json")
    logger.info(
        "Validation passed (%d original, %d synthetic rows, %d columns). Saved %s",
        len(data.original),
        len(data.synthetic),
        len(data.schema.columns),
        validation_path,
    )

    n_before = len(data.synthetic)
    keep = np.arange(n_before)
    reduction: dict[str, Any] = {
        "reduction_method": opts.reduction_method,
        "random_seed": opts.random_seed,
        "n_original": len(data.original),
        "n_synthetic_before": n_before,
    }

    if opts.remove_exact_matches == "on":
        (g_o, g_s), n_groups = record_group_ids(
            [data.original, data.synthetic], data.schema.columns
        )
        in_orig = np.bincount(g_o, minlength=n_groups)[g_s] > 0
        keep = keep[~in_orig]
        reduction["exact_matches_removed"] = int(in_orig.sum())
        logger.info(
            "Removed %d synthetic records identical to original records.", int(in_orig.sum())
        )

    target = len(data.original)
    inference_subset = None
    if len(keep) > target and opts.reduction_method:
        options: dict[str, Any] = {
            "model": opts.model,
            "model_params": opts.model_params,
            "distance_metric": opts.distance_metric,
            "use_gpu": opts.use_gpu,
            "n_jobs": opts.n_jobs,
            "target_margin": opts.target_margin,
            "original_path": opts.original,
            "synthetic_path": opts.synthetic,
            "output_dir": opts.output_dir,
        }
        if opts.reduction_method == "inference_distance":
            options["inference_threshold"], reduction["threshold_origin"] = (
                _resolve_inference_threshold(opts)
            )
        ctx = ReductionContext(
            original=data.original,
            synthetic=data.synthetic.iloc[keep].reset_index(drop=True),
            synthetic_full=data.synthetic,
            rows=keep,
            schema=data.schema,
            target_size=target,
            random_seed=opts.random_seed,
            options=options,
        )
        logger.info(
            "Reducing synthetic data from %d to %d rows with '%s'.",
            len(keep),
            target,
            opts.reduction_method,
        )
        result = reduce_records(opts.reduction_method, ctx)
        deleted_rows = keep[result.delete_positions]
        keep = np.delete(keep, result.delete_positions)
        reduction["strategy_details"] = result.details
        reduction["n_deleted"] = int(len(deleted_rows))
        reduction["deleted_row_indices"] = deleted_rows
        if result.inference_details is not None:
            inference_subset = result.inference_details.subset(
                np.setdiff1d(np.arange(len(ctx.synthetic)), result.delete_positions)
            )
    elif len(keep) > target:
        logger.warning(
            "Synthetic data (%d rows) is larger than original (%d rows) but no --reduction_method "
            "was given; rows are kept as they are.",
            len(keep),
            target,
        )
    elif opts.reduction_method:
        logger.info("Synthetic data is not larger than original; no reduction needed.")

    # 원자료 표기를 그대로 보존해 저장 (열 순서만 원본 기준으로 정렬)
    text = io_utils.read_table_as_strings(opts.synthetic)[list(data.synthetic_raw.columns)]
    processed_path = io_utils.write_csv(
        text.iloc[keep].reset_index(drop=True), output_path_for(opts.output_dir)
    )
    reduction.update(
        {
            "n_synthetic_after": int(len(keep)),
            "n_deleted_total": int(n_before - len(keep)),
            "processed_synthetic": processed_path.as_posix(),
        }
    )

    if inference_subset is not None:
        key = inference_cache_key(opts.original, processed_path, data.schema, opts.distance_metric)
        store_inference_cache(
            opts.output_dir,
            key,
            inference_subset,
            opts.distance_metric,
            source="preprocess (processed synthetic)",
        )
        reduction["processed_inference_risk"] = inference_subset.value
        reduction["processed_cache_key"] = key

    reduction["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    reduction_path = io_utils.save_json({**reduction, **common}, out_dir / "reduction.json")
    logger.info("Processed synthetic data (%d rows) saved to %s", len(keep), processed_path)
    return {
        "validation_file": validation_path.as_posix(),
        "reduction_file": reduction_path.as_posix(),
        "processed_synthetic": processed_path.as_posix(),
        "reduction": reduction,
    }
