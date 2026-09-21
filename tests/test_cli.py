"""CLI 통합 테스트 (test_data/adult_*.csv 사용)."""

from __future__ import annotations

import pytest
import yaml
from conftest import ORIGIN_CSV, SYNTH_CSV

from sdqe import io_utils
from sdqe.cli import main

ORIG, SYN = str(ORIGIN_CSV), str(SYNTH_CSV)


def test_threshold_then_preprocess_then_cached_metric(tmp_path):
    out = str(tmp_path)
    assert (
        main(
            [
                "threshold",
                "--metric",
                "inference_risk",
                "--original",
                ORIG,
                "--n_simulations",
                "5",
                "--output_dir",
                out,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "preprocess",
                "--original",
                ORIG,
                "--synthetic",
                SYN,
                "--output_dir",
                out,
                "--reduction_method",
                "inference_distance",
            ]
        )
        == 0
    )
    reduction = io_utils.load_json(tmp_path / "preprocess" / "reduction.json")
    assert reduction["n_synthetic_before"] == 700 and reduction["n_synthetic_after"] == 500
    processed = reduction["processed_synthetic"]
    assert (
        main(
            [
                "metric",
                "--name",
                "inference_risk",
                "--original",
                ORIG,
                "--synthetic",
                processed,
                "--output_dir",
                out,
            ]
        )
        == 0
    )
    result = io_utils.load_json(tmp_path / "metrics" / "inference_risk.json")
    assert result["cache_hit"] is True
    assert result["value"] == pytest.approx(reduction["processed_inference_risk"])
    assert result["details"]["n_evaluated"] == 500 - result["details"]["n_ties"]
    for key in (
        "threshold",
        "threshold_source",
        "passed",
        "parameters",
        "input_info",
        "computed_at",
    ):
        assert key in result


def test_processed_file_preserves_original_text(tmp_path):
    main(
        [
            "preprocess",
            "--original",
            ORIG,
            "--synthetic",
            SYN,
            "--output_dir",
            str(tmp_path),
            "--reduction_method",
            "random",
        ]
    )
    processed = io_utils.read_table_as_strings(tmp_path / "preprocess" / "synthetic_processed.csv")
    source = io_utils.read_table_as_strings(SYN)
    deleted = io_utils.load_json(tmp_path / "preprocess" / "reduction.json")["deleted_row_indices"]
    kept = source.drop(index=deleted).reset_index(drop=True)
    assert processed.equals(kept)


def test_distance_metric_mismatch_is_rejected(tmp_path, capsys):
    out = str(tmp_path)
    main(
        [
            "threshold",
            "--metric",
            "inference_risk",
            "--original",
            ORIG,
            "--n_simulations",
            "3",
            "--distance_metric",
            "cosine",
            "--output_dir",
            out,
        ]
    )
    code = main(
        [
            "metric",
            "--name",
            "inference_risk",
            "--original",
            ORIG,
            "--synthetic",
            SYN,
            "--distance_metric",
            "gower",
            "--output_dir",
            out,
        ]
    )
    assert code == 2 and "distance_metric mismatch" in capsys.readouterr().out


def test_unsupported_threshold_method_message(tmp_path, capsys):
    code = main(
        [
            "threshold",
            "--metric",
            "pmse",
            "--method",
            "distribution",
            "--original",
            ORIG,
            "--output_dir",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "--method distribution is not supported for metric 'pmse'" in capsys.readouterr().out
    assert (
        main(
            [
                "threshold",
                "--metric",
                "linkage_risk",
                "--original",
                ORIG,
                "--output_dir",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_linkage_metric_and_threshold_range(tmp_path):
    base = [
        "metric",
        "--name",
        "linkage_risk",
        "--original",
        ORIG,
        "--synthetic",
        SYN,
        "--output_dir",
        str(tmp_path),
        "--quasi_identifiers",
        "gender",
        "race",
        "--sensitive_attributes",
        "income",
    ]
    assert main(base) == 0
    res = io_utils.load_json(tmp_path / "metrics" / "linkage_risk.json")
    assert res["threshold"] == 0.7 and res["threshold_source"] == "user_defined"
    assert main(base + ["--linkage_threshold", "1.0"]) == 2


def test_report_with_config_and_cli_override(tmp_path):
    cfg = {
        "original": ORIG,
        "synthetic": SYN,
        "output_dir": str(tmp_path / "wrong"),
        "n_simulations": 3,
        "reduction_method": "pmse_probability",
        "model": "logistic_regression",
        "plots": "off",
        "quasi_identifiers": ["gender", "race"],
        "sensitive_attributes": ["income"],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    assert main(["report", "--config", str(path), "--output_dir", str(tmp_path / "out")]) == 0
    report = io_utils.load_json(tmp_path / "out" / "report.json")
    assert set(report["metrics"]) == {
        "univariate_similarity",
        "bivariate_similarity",
        "pmse",
        "single_out_risk",
        "inference_risk",
        "linkage_risk",
    }
    assert report["preprocess"]["n_synthetic_after"] == 500
    assert not (tmp_path / "wrong").exists()


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("original: a.csv\nno_such_option: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no_such_option"):
        main(["utility", "--config", str(path)])


def test_group_skip_metric(tmp_path):
    assert (
        main(
            [
                "utility",
                "--original",
                ORIG,
                "--synthetic",
                SYN,
                "--output_dir",
                str(tmp_path),
                "--skip_metrics",
                "bivariate_similarity",
                "--plots",
                "off",
            ]
        )
        == 0
    )
    assert not (tmp_path / "metrics" / "bivariate_similarity.json").exists()
    assert (tmp_path / "metrics" / "pmse.json").exists()
