"""linkage_risk (CAP) 계산 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ORIGIN_CSV, SYNTH_CSV

from sdqe import io_utils
from sdqe.metrics.privacy.linkage_risk import (
    CSV_NA_VALUES,
    calculate_cap,
    compute_linkage_risk,
    validate_linkage_threshold,
)


@pytest.fixture(scope="module")
def string_data():
    return (
        io_utils.read_table_as_strings(ORIGIN_CSV, CSV_NA_VALUES),
        io_utils.read_table_as_strings(SYNTH_CSV, CSV_NA_VALUES),
    )


def test_cap_handles_missing_values_and_korean_names():
    rng = np.random.default_rng(0)
    n = 300

    def frame(seed):
        r = np.random.default_rng(seed)
        df = pd.DataFrame(
            {
                "나이 구간": r.choice(["20", "30", "40", ""], n),
                "성별(코드)": r.choice(["M", "F", "None"], n),
                "소득 등급": r.choice(["1", "2", "3", "nan"], n),
                "질병": r.choice(["A", "B", "null"], n),
            }
        ).astype("string")
        return df.mask(pd.DataFrame(r.random((n, 4)) < 0.05, columns=df.columns))

    orig, syn = frame(1), frame(2)
    del rng
    keys, targets = ["나이 구간", "성별(코드)"], ["소득 등급", "질병"]
    for mode in ("individual", "combined"):
        records, summary = calculate_cap(orig, syn, keys, targets, 0.5, "exclude", mode)
        evaluated = records[records["evaluated"]]
        skipped = records[~records["evaluated"]]
        # 결측이 있는 행은 사유와 함께 제외되고, 평가된 행의 CAP 은 항상 0~1 이다.
        assert len(evaluated) > 0
        assert evaluated["cap"].between(0, 1).all()
        assert skipped["cap"].isna().all()
        assert skipped["exclusion_reason"].notna().all()
        assert summary["mean_cap"].between(0, 1).all()
        assert set(summary["evaluation_type"]) <= {"individual", "combined"}


def test_denominator_zero_gives_cap_zero():
    orig = pd.DataFrame({"k": ["a", "b"], "t": ["x", "y"]}).astype("string")
    syn = pd.DataFrame({"k": ["a", "a"], "t": ["x", "z"]}).astype("string")
    records, summary = calculate_cap(orig, syn, ["k"], ["t"])
    assert records["cap"].tolist() == [0.5, 0.0]
    assert summary.loc[0, "unmatched_count"] == 1
    assert summary.loc[0, "mean_cap"] == pytest.approx(0.25)


def test_linkage_threshold_range(caplog):
    with pytest.raises(ValueError):
        validate_linkage_threshold(1.0)
    with pytest.raises(ValueError):
        validate_linkage_threshold(-0.1)
    validate_linkage_threshold(0.3)
    assert "below 0.5" in caplog.text


def test_compute_linkage_risk_summary(string_data):
    orig, syn = string_data
    result, records = compute_linkage_risk(orig, syn, ["gender", "race"], ["income"], 0.7)
    assert result["threshold_source"] == "user_defined"
    assert result["exceed_count"] == int(records["exceeds_threshold"].sum())
    assert result["passed"] == (result["exceed_count"] == 0)
