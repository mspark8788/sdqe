"""``--data_dir``: 폴더에서 ``<prefix>_origin`` / ``<prefix>_synth`` 쌍을 찾아 평가합니다."""

from __future__ import annotations

import os

import pytest
import yaml
from conftest import make_mixed

from sdqe import io_utils
from sdqe.cli import main
from sdqe.data_dir import DataDirError, find_data_pair


def _write_pair(folder, prefix, ext=".csv", synth=True, n=(60, 80)):
    folder.mkdir(parents=True, exist_ok=True)
    make_mixed(n[0], 1).to_csv(folder / f"{prefix}_origin{ext}", index=False)
    if synth:
        make_mixed(n[1], 2).to_csv(folder / f"{prefix}_synth{ext}", index=False)


# ---------------------------------------------------------------- 탐색 규칙
def test_single_pair_is_found_automatically(tmp_path):
    _write_pair(tmp_path, "adult")
    pair = find_data_pair(tmp_path)
    assert pair.prefix == "adult"
    assert pair.original.name == "adult_origin.csv"
    assert pair.synthetic.name == "adult_synth.csv"
    assert pair.output_dir == tmp_path / "adult_results"


def test_multiple_pairs_require_prefix(tmp_path):
    _write_pair(tmp_path, "adult")
    _write_pair(tmp_path, "bank")
    with pytest.raises(DataDirError, match="'adult', 'bank'.*--prefix"):
        find_data_pair(tmp_path)
    assert find_data_pair(tmp_path, prefix="bank").original.name == "bank_origin.csv"


def test_missing_synthetic_is_reported(tmp_path):
    _write_pair(tmp_path, "adult", synth=False)
    with pytest.raises(DataDirError, match="Missing synthetic file for: adult"):
        find_data_pair(tmp_path)
    # threshold 는 원본만 있으면 됩니다.
    pair = find_data_pair(tmp_path, need_synthetic=False)
    assert pair.synthetic is None and pair.output_dir == tmp_path / "adult_results"


def test_unknown_prefix(tmp_path):
    _write_pair(tmp_path, "adult")
    with pytest.raises(DataDirError, match="No 'bank_origin"):
        find_data_pair(tmp_path, prefix="bank")


def test_same_role_in_two_formats_is_ambiguous(tmp_path):
    _write_pair(tmp_path, "adult")
    (tmp_path / "adult_origin.parquet").write_bytes(b"")  # 탐색은 이름만 봅니다
    with pytest.raises(DataDirError, match="Ambiguous original files"):
        find_data_pair(tmp_path)


def test_other_formats_case_and_lock_files(tmp_path):
    (tmp_path / "survey_ORIGIN.parquet").write_bytes(b"")  # 탐색은 이름만 봅니다
    (tmp_path / "survey_Synth.xlsx").write_bytes(b"")
    (tmp_path / "~$survey_origin.xlsx").write_bytes(b"lock")  # Excel 잠금 파일
    (tmp_path / "notes_origin.txt").write_text("x")  # 지원하지 않는 확장자
    pair = find_data_pair(tmp_path)
    assert pair.prefix == "survey"
    assert pair.original.suffix == ".parquet" and pair.synthetic.suffix == ".xlsx"


def test_not_a_directory(tmp_path):
    with pytest.raises(DataDirError, match="is not a directory"):
        find_data_pair(tmp_path / "nope")


# ---------------------------------------------------------------- CLI
def test_threshold_then_metric_share_results_folder(tmp_path):
    _write_pair(tmp_path, "adult")
    base = ["--data_dir", str(tmp_path)]
    assert main(["threshold", "--metric", "single_out_risk", "--n_simulations", "3", *base]) == 0
    assert main(["metric", "--name", "single_out_risk", *base]) == 0
    out = tmp_path / "adult_results"
    assert (out / "threshold" / "single_out_risk.json").is_file()
    result = io_utils.load_json(out / "metrics" / "single_out_risk.json")
    assert result["threshold_source"] == "simulation" and result["passed"] is not None


def test_metric_uses_preprocessed_file_when_present(tmp_path, capsys):
    _write_pair(tmp_path, "adult")
    base = ["--data_dir", str(tmp_path)]
    assert main(["preprocess", "--reduction_method", "random", *base]) == 0
    capsys.readouterr()
    assert main(["metric", "--name", "single_out_risk", *base]) == 0
    log = capsys.readouterr().out
    assert "synthetic_processed.csv" in log
    result = io_utils.load_json(tmp_path / "adult_results" / "metrics" / "single_out_risk.json")
    assert result["input_info"]["n_synthetic"] == 60  # 80 -> 60 으로 축소된 파일


def test_stale_preprocessed_file_falls_back_to_raw(tmp_path, capsys):
    _write_pair(tmp_path, "adult")
    base = ["--data_dir", str(tmp_path)]
    assert main(["preprocess", "--reduction_method", "random", *base]) == 0
    synth = tmp_path / "adult_synth.csv"
    later = (tmp_path / "adult_results" / "preprocess" / "synthetic_processed.csv").stat().st_mtime
    os.utime(synth, (later + 10, later + 10))  # 전처리 후에 합성 파일이 바뀐 상황
    capsys.readouterr()
    assert main(["metric", "--name", "single_out_risk", *base]) == 0
    assert "Re-run preprocess" in capsys.readouterr().out
    result = io_utils.load_json(tmp_path / "adult_results" / "metrics" / "single_out_risk.json")
    assert result["input_info"]["n_synthetic"] == 80


def test_explicit_output_dir_wins(tmp_path):
    _write_pair(tmp_path, "adult")
    out = tmp_path / "elsewhere"
    assert (
        main(
            [
                "metric",
                "--name",
                "single_out_risk",
                "--data_dir",
                str(tmp_path),
                "--output_dir",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "metrics" / "single_out_risk.json").is_file()
    assert not (tmp_path / "adult_results").exists()


def test_folder_errors_exit_with_code_2(tmp_path, capsys):
    _write_pair(tmp_path, "adult")
    _write_pair(tmp_path, "bank")
    assert main(["metric", "--name", "single_out_risk", "--data_dir", str(tmp_path)]) == 2
    assert "--prefix" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["metric", "--name", "pmse", "--data_dir", "d", "--original", "a.csv"],
        ["metric", "--name", "pmse", "--prefix", "adult", "--original", "a", "--synthetic", "b"],
        ["metric", "--name", "pmse"],
        ["threshold", "--metric", "pmse"],
    ],
)
def test_invalid_data_source_combinations(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_cli_data_dir_overrides_config_files(tmp_path):
    _write_pair(tmp_path, "adult")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump({"original": "missing.csv", "synthetic": "missing.csv", "plots": "off"}),
        encoding="utf-8",
    )
    argv = ["utility", "--config", str(cfg), "--data_dir", str(tmp_path)]
    assert main([*argv, "--metrics", "univariate_similarity"]) == 0
    assert (tmp_path / "adult_results" / "metrics" / "univariate_similarity.json").is_file()


# ---------------------------------------------------------------- 입력 옆 전처리본 사본
def test_preprocess_writes_processed_copy_next_to_input(tmp_path):
    _write_pair(tmp_path, "adult")
    assert main(["preprocess", "--reduction_method", "random", "--data_dir", str(tmp_path)]) == 0
    copy = tmp_path / "adult_synth_processed.csv"
    inner = tmp_path / "adult_results" / "preprocess" / "synthetic_processed.csv"
    assert copy.read_bytes() == inner.read_bytes()
    reduction = io_utils.load_json(tmp_path / "adult_results" / "preprocess" / "reduction.json")
    assert reduction["processed_synthetic_copy"] == copy.as_posix()
    # 사본이 생겨도 폴더 탐색 결과는 그대로입니다 (_processed 는 _synth 로 끝나지 않음).
    assert find_data_pair(tmp_path).synthetic.name == "adult_synth.csv"


def test_processed_copy_with_explicit_paths(tmp_path):
    src = tmp_path / "inputs"
    _write_pair(src, "survey")
    argv = [
        "preprocess",
        "--original",
        str(src / "survey_origin.csv"),
        "--synthetic",
        str(src / "survey_synth.csv"),
        "--output_dir",
        str(tmp_path / "out"),
    ]
    assert main(argv) == 0
    assert (src / "survey_synth_processed.csv").is_file()


def test_processed_copy_failure_only_warns(tmp_path, capsys):
    _write_pair(tmp_path, "adult")
    (tmp_path / "adult_synth_processed.csv").mkdir()  # 이 이름으로 파일을 쓸 수 없는 상황
    assert main(["preprocess", "--data_dir", str(tmp_path)]) == 0
    assert "Could not write adult_synth_processed.csv" in capsys.readouterr().out
    reduction = io_utils.load_json(tmp_path / "adult_results" / "preprocess" / "reduction.json")
    assert reduction["processed_synthetic_copy"] is None
