"""안전성 지표: single_out_risk (식 1, 2), inference_risk (식 4) 와 캐시."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ORIGIN_CSV, SYNTH_CSV, make_mixed
from test_distance import naive_matrix

from sdqe import io_utils
from sdqe.common.distance import prepare_features
from sdqe.metrics.privacy.inference_risk import (
    compute_inference_details,
    load_or_compute_inference_details,
    summarize_inference,
)
from sdqe.metrics.privacy.single_out_risk import compute_single_out_risk
from sdqe.preprocess.validation import Schema, validate_pair


# ---------------------------------------------------------------- single_out_risk
def test_single_out_equation_1_is_match_ratio():
    """식 (1) 은 원본과 완전히 일치하는 합성 레코드의 단순 비율이다."""
    orig = io_utils.read_table(ORIGIN_CSV)
    syn = pd.concat([io_utils.read_table(SYNTH_CSV), orig.iloc[:7]], ignore_index=True)
    data = validate_pair(orig, syn)
    ours = compute_single_out_risk(
        data.original, data.synthetic, data.schema, duplicate_adjustment=False
    )
    keys = set(map(tuple, data.original.astype(str).to_numpy()))
    matched = sum(tuple(row) in keys for row in data.synthetic.astype(str).to_numpy())
    assert ours["n_matched_records"] == matched >= 7
    assert ours["value"] == pytest.approx(matched / len(data.synthetic), abs=0)


def test_single_out_equation_2_divides_by_duplicates():
    schema = Schema(("x", "y"), ("x", "y"), ())
    orig = pd.DataFrame({"x": ["a", "a", "a", "b", "c"], "y": ["1", "1", "1", "2", "3"]})
    syn = pd.DataFrame({"x": ["a", "b", "z", "c"], "y": ["1", "2", "9", "3"]})
    # 항: a1 -> 1/3, b2 -> 1/1, z9 -> 0, c3 -> 1/1
    r2 = compute_single_out_risk(orig, syn, schema, duplicate_adjustment=True)
    r1 = compute_single_out_risk(orig, syn, schema, duplicate_adjustment=False)
    assert r2["value"] == pytest.approx((1 / 3 + 1 + 0 + 1) / 4)
    assert r1["value"] == pytest.approx(3 / 4)


def test_single_out_treats_missing_as_equal():
    schema = Schema(("x", "y"), ("x",), ("y",))
    orig = pd.DataFrame({"x": ["a", np.nan], "y": [1.0, np.nan]})
    syn = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
    assert compute_single_out_risk(orig, syn, schema)["value"] == 1.0


# ---------------------------------------------------------------- inference_risk
def naive_inference(original, synthetic, schema, metric):
    """식 (4) 를 정의대로 계산 (전체 거리 행렬 + 루프). 반환: (지시함수값, 동률 여부)."""
    fo, fs = prepare_features(original, [synthetic], schema, metric)
    d_so = naive_matrix(fs, fo)
    d_oo = naive_matrix(fo, fo)
    np.fill_diagonal(d_oo, np.inf)
    flags, ties = [], []
    for i in range(fs.n_rows):
        k = int(np.argmin(d_so[i]))  # 최근접이 여러 개면 가장 작은 행 번호
        d1, d2 = d_so[i, k], d_oo[k].min()
        flags.append(int(d1 < d2))
        ties.append(d1 == d2)
    return np.array(flags), np.array(ties)


def naive_inference_value(original, synthetic, schema, metric):
    """동률을 분모에서 제외한 추론위험도 (참고문헌 [1] 안내서 68쪽)."""
    flags, ties = naive_inference(original, synthetic, schema, metric)
    return flags.sum() / (len(flags) - ties.sum())


@pytest.mark.parametrize("metric", ["gower", "cosine"])
def test_inference_matches_naive(metric):
    data = validate_pair(make_mixed(50, 11), make_mixed(35, 12))
    details = compute_inference_details(
        data.original, data.synthetic, data.schema, metric, n_jobs=2
    )
    flags, ties = naive_inference(data.original, data.synthetic, data.schema, metric)
    np.testing.assert_array_equal(details.flags, flags)
    np.testing.assert_array_equal(details.ties, ties)
    assert details.n_evaluated == len(flags) - ties.sum()
    assert details.value == pytest.approx(flags.sum() / (len(flags) - ties.sum()))


def test_inference_summary_excludes_ties_from_denominator(mixed_pair):
    d = compute_inference_details(mixed_pair.original, mixed_pair.synthetic, mixed_pair.schema)
    s = summarize_inference(d)
    assert s["reference_value"] == 0.5
    assert s["n_evaluated"] == len(d.flags) - s["n_ties"]
    assert s["value"] == pytest.approx(s["n_flagged"] / s["n_evaluated"])
    # Originality 는 이 지표의 옛 이름이므로 1 - value 같은 파생값을 따로 두지 않습니다.
    assert "derived" not in s


def test_inference_cache_roundtrip(tmp_path):
    orig, syn = tmp_path / "o.csv", tmp_path / "s.csv"
    make_mixed(30, 1).to_csv(orig, index=False)
    make_mixed(25, 2).to_csv(syn, index=False)
    data = validate_pair(io_utils.read_table(orig), io_utils.read_table(syn))
    args = (data.original, data.synthetic, data.schema, "gower", orig, syn, tmp_path)
    first, hit1, key1 = load_or_compute_inference_details(*args)
    second, hit2, key2 = load_or_compute_inference_details(*args)
    assert (hit1, hit2) == (False, True) and key1 == key2
    np.testing.assert_array_equal(first.flags, second.flags)
    np.testing.assert_allclose(first.reference_distance, second.reference_distance)
    # 다른 거리 방식은 다른 캐시
    _, hit3, key3 = load_or_compute_inference_details(*args[:3], "cosine", *args[4:])
    assert not hit3 and key3 != key1
