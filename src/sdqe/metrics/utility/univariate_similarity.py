"""1차원 유사성 ``univariate_similarity`` (명세 5.1절).

변수별로 원본과 합성의 분포를 비교합니다. 임계값 계산 대상이 아니며
검정 통계량과 p-value 를 그대로 보고합니다.

- 수치형: 2표본 Kolmogorov-Smirnov 검정 (``scipy.stats.ks_2samp``), ECDF plot
- 범주형: 카이제곱 동질성 검정 (``scipy.stats.chi2_contingency`` 기본 설정), 비율 막대 plot

각 plot 아래에 검정 결과를 텍스트로 적고, 전체 변수를 한 장에 담은 총정리 plot 을 추가로 만듭니다.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from sdqe import config
from sdqe.common import plotting
from sdqe.preprocess.validation import Schema

logger = logging.getLogger("sdqe")


def compare_numeric(original: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    """수치형 변수의 KS 검정 결과 (결측 제외)."""
    o, s = original.dropna(), synthetic.dropna()
    if not len(o) or not len(s):
        return {"test": "ks", "statistic": None, "p_value": None, "note": "no non-missing values"}
    with warnings.catch_warnings():
        # 동률이 많으면 scipy 가 정확 계산 대신 점근 근사로 바꾸며 경고합니다 (결과는 유효).
        warnings.simplefilter("ignore", RuntimeWarning)
        res = ks_2samp(o.to_numpy(dtype=float), s.to_numpy(dtype=float))
    return {"test": "ks", "statistic": float(res.statistic), "p_value": float(res.pvalue)}


def compare_categorical(original: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    """범주형 변수의 카이제곱 검정 결과 (결측 제외, 범주 합집합 기준 빈도표)."""
    o = original.dropna().value_counts()
    s = synthetic.dropna().value_counts()
    categories = sorted(set(o.index) | set(s.index), key=str)
    table = [[int(o.get(c, 0)) for c in categories], [int(s.get(c, 0)) for c in categories]]
    try:
        stat, p, dof, _ = chi2_contingency(table)
    except ValueError as exc:
        return {"test": "chi_square", "statistic": None, "p_value": None, "note": str(exc)}
    return {
        "test": "chi_square",
        "statistic": float(stat),
        "p_value": float(p),
        "dof": int(dof),
        "n_categories": len(categories),
    }


def format_test(result: dict[str, Any]) -> str:
    """plot 아래에 적을 검정 결과 문구."""
    name = "KS test" if result["test"] == "ks" else "Chi-square test"
    if result.get("statistic") is None:
        return f"{name}: not available ({result.get('note', '')})"
    return f"{name}: statistic={result['statistic']:.4f}, p-value={result['p_value']:.4g}"


def _plot_variable(
    ax,
    original: pd.Series,
    synthetic: pd.Series,
    is_cat: bool,
    title: str,
    label_len: int,
    label_fontsize: float | None = None,
) -> None:
    if is_cat:
        plotting.plot_category_bars(ax, original, synthetic, title, label_len, label_fontsize)
    else:
        plotting.plot_ecdf(ax, original, synthetic, title)


def save_variable_figure(
    column: str, original: pd.Series, synthetic: pd.Series, is_cat: bool, text: str, path: Path
) -> None:
    """변수 하나의 plot 을 그리고 아래에 검정 결과를 적어 저장합니다."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plotting.setup_korean_font()
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_variable(ax, original, synthetic, is_cat, str(column), 20)
    ax.legend()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_summary_figure(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
    results: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """전체 변수를 한 장에 담은 총정리 plot 을 저장합니다.

    변수 수에 따라 행/열 배치와 글자 크기, 여백을 조정합니다.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plotting.setup_korean_font()
    columns = list(schema.columns)
    n_rows, n_cols = plotting.grid_shape(len(columns))
    scale = 1.0 if len(columns) <= 9 else (0.85 if len(columns) <= 25 else 0.7)
    fontsize = 10 * scale
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)
    for ax, col in zip(axes.ravel(), columns):
        _plot_variable(
            ax,
            original[col],
            synthetic[col],
            schema.is_categorical(col),
            plotting.shorten(col, 30),
            10,
            fontsize * 0.8,
        )
        ax.title.set_fontsize(fontsize + 1)
        ax.yaxis.label.set_fontsize(fontsize * 0.9)
        ax.tick_params(axis="y", labelsize=fontsize * 0.8)
        res = results[col]
        stat = "n/a" if res.get("statistic") is None else f"{res['statistic']:.3f}"
        pval = "n/a" if res.get("p_value") is None else f"{res['p_value']:.3g}"
        label = "KS" if res["test"] == "ks" else "Chi2"
        ax.text(
            0.5,
            -0.32 if schema.is_categorical(col) else -0.18,
            f"{label}={stat}, p={pval}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fontsize * 0.9,
        )
    for ax in axes.ravel()[len(columns) :]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=fontsize)
    fig.suptitle("univariate_similarity summary (Original vs Synthetic)", fontsize=fontsize + 4)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5 * scale + 1.0)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def compute_univariate_similarity(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
    figures_dir: str | Path | None = None,
    alpha: float = config.DEFAULT_ALPHA,
) -> dict[str, Any]:
    """변수별 분포 비교 검정을 하고 plot 을 저장합니다.

    Args:
        original: 원본 데이터 (타입 정리 완료).
        synthetic: 합성 데이터 (타입 정리 완료).
        schema: 분석 스키마.
        figures_dir: plot 저장 디렉터리. ``None`` 이면 plot 을 그리지 않습니다.
        alpha: 요약에 쓰는 유의수준 (판정용이 아니라 보고용).

    Returns:
        변수별 검정 결과와 요약을 담은 딕셔너리. 단일 지표값이 없으므로 ``value`` 는 None.
    """
    results: dict[str, dict[str, Any]] = {}
    for col in schema.columns:
        is_cat = schema.is_categorical(col)
        res = (compare_categorical if is_cat else compare_numeric)(original[col], synthetic[col])
        res["type"] = "categorical" if is_cat else "numeric"
        results[col] = res

    figures: dict[str, str] = {}
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        for col in schema.columns:
            name = plotting.sanitize_filename(col)
            while name in used_names:  # 치환 후 파일명이 겹치면 번호를 붙임
                name += "_"
            used_names.add(name)
            path = figures_dir / f"{config.UNIVARIATE_SIMILARITY}_{name}.png"
            save_variable_figure(
                col,
                original[col],
                synthetic[col],
                schema.is_categorical(col),
                format_test(results[col]),
                path,
            )
            results[col]["figure"] = path.as_posix()
            figures[col] = path.as_posix()
        summary_path = figures_dir / f"{config.UNIVARIATE_SIMILARITY}_summary.png"
        save_summary_figure(original, synthetic, schema, results, summary_path)
        figures["summary"] = summary_path.as_posix()
        logger.info("Saved %d univariate figures to %s", len(figures), figures_dir)

    p_values = [r["p_value"] for r in results.values() if r.get("p_value") is not None]
    return {
        "value": None,
        "alpha": alpha,
        "n_variables": len(results),
        "n_significant": int(np.sum(np.asarray(p_values) < alpha)) if p_values else 0,
        "variables": results,
        "figures": figures,
    }
