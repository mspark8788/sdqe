"""한글 폰트 설정과 공용 plot 유틸.

plot 을 그리는 코드는 그리기 전에 :func:`setup_korean_font` 를 먼저 호출합니다.
"""

from __future__ import annotations

import logging
import math
import re

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

logger = logging.getLogger("sdqe")

KOREAN_FONT_CANDIDATES = (
    "Malgun Gothic",
    "AppleGothic",
    "Apple SD Gothic Neo",
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Source Han Sans KR",
    "UnDotum",
)
FALLBACK_FONT = "DejaVu Sans"
ORIGINAL_COLOR = "#1f77b4"
SYNTHETIC_COLOR = "#ff7f0e"

_font_state: dict[str, str | None] = {}


def setup_korean_font() -> str | None:
    """설치된 한글 폰트를 찾아 matplotlib 기본 폰트로 설정합니다.

    찾지 못하면 경고를 남기고 깨지지 않는 대체 폰트(DejaVu Sans)로 진행합니다.
    여러 번 호출해도 폰트 탐색은 한 번만 합니다.

    Returns:
        설정한 한글 폰트 이름. 찾지 못하면 ``None``.
    """
    if "font" not in _font_state:
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((name for name in KOREAN_FONT_CANDIDATES if name in available), None)
        if chosen is None:
            logger.warning(
                "No Korean font found (tried %s). Korean labels may not render; using %s.",
                ", ".join(KOREAN_FONT_CANDIDATES),
                FALLBACK_FONT,
            )
        _font_state["font"] = chosen
    chosen = _font_state["font"]
    plt.rcParams["font.family"] = [chosen, FALLBACK_FONT] if chosen else [FALLBACK_FONT]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """변수명을 파일명으로 쓸 수 있게 바꿉니다. 한글은 유지하고 경로 예약 문자만 치환합니다."""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name)).strip("._")
    return (cleaned or "column")[:max_length]


def grid_shape(n_items: int, max_cols: int = 6) -> tuple[int, int]:
    """항목 수에 맞는 (행, 열) 배치를 계산합니다."""
    n_cols = max(1, min(max_cols, math.ceil(math.sqrt(n_items))))
    n_rows = max(1, math.ceil(n_items / n_cols))
    return n_rows, n_cols


def shorten(label: object, max_len: int) -> str:
    """긴 라벨을 잘라 말줄임표를 붙입니다."""
    text = str(label)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def plot_ecdf(ax: plt.Axes, original: pd.Series, synthetic: pd.Series, title: str) -> None:
    """수치형 변수의 원본/합성 경험적 누적분포(ECDF)를 계단 그래프로 그립니다."""
    o = np.sort(original.dropna().to_numpy(dtype=float))
    s = np.sort(synthetic.dropna().to_numpy(dtype=float))
    if len(o):
        x = np.linspace(o.min(), o.max(), 50) if o.max() > o.min() else np.array([o.min()])
        ax.step(
            x,
            np.searchsorted(o, x, side="right") / len(o),
            where="post",
            label="Original",
            color=ORIGINAL_COLOR,
        )
        if len(s):
            ax.step(
                x,
                np.searchsorted(s, x, side="right") / len(s),
                where="post",
                label="Synthetic",
                color=SYNTHETIC_COLOR,
            )
    ax.set_title(title)
    ax.set_ylabel("ECDF")
    ax.set_ylim(-0.02, 1.02)


def plot_category_bars(
    ax: plt.Axes,
    original: pd.Series,
    synthetic: pd.Series,
    title: str,
    max_label_len: int = 20,
    label_fontsize: float | None = None,
) -> None:
    """범주형 변수의 원본/합성 범주 비율(%)을 묶은 막대그래프로 그립니다."""
    o = original.dropna().value_counts(normalize=True) * 100
    s = synthetic.dropna().value_counts(normalize=True) * 100
    categories = sorted(set(o.index) | set(s.index), key=str)
    x = np.arange(len(categories))
    width = 0.4
    ax.bar(
        x - width / 2,
        [o.get(c, 0.0) for c in categories],
        width,
        label="Original",
        color=ORIGINAL_COLOR,
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        [s.get(c, 0.0) for c in categories],
        width,
        label="Synthetic",
        color=SYNTHETIC_COLOR,
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [shorten(c, max_label_len) for c in categories],
        rotation=45,
        ha="right",
        fontsize=label_fontsize,
    )
    ax.set_title(title)
    ax.set_ylabel("Percentage (%)")
