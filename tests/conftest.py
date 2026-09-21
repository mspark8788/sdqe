from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # conftest 헬퍼를 이름으로 import 하기 위함

TEST_DATA = ROOT.parent / "test_data"
ORIGIN_CSV = TEST_DATA / "adult_origin.csv"
SYNTH_CSV = TEST_DATA / "adult_synth.csv"


def make_mixed(n: int, seed: int, with_names: bool = False) -> pd.DataFrame:
    """동률과 중복이 많은 혼합형 데이터 (한글/공백/특수문자 열 이름 옵션)."""
    rng = np.random.default_rng(seed)
    names = (
        ["나이 (세)", "소득/월", "성별", "지역-코드", "등급"]
        if with_names
        else ["age", "income", "sex", "region", "grade"]
    )
    return pd.DataFrame(
        {
            names[0]: rng.integers(20, 25, n),
            names[1]: rng.choice([100.0, 150.0, 200.0, 250.0], n),
            names[2]: rng.choice(["M", "F"], n).astype(object),
            names[3]: rng.choice(["서울", "부산", "대구"], n).astype(object),
            names[4]: rng.choice(["a", "b", "c", "d"], n).astype(object),
        }
    )


@pytest.fixture
def mixed_pair():
    from sdqe.preprocess.validation import validate_pair

    return validate_pair(make_mixed(60, 1), make_mixed(45, 2))
