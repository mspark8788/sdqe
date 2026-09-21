"""pMSE (propensity score Mean Squared Error, 참고문헌 [7] Woo et al., 2009).

원본과 합성을 합친 뒤 "합성 여부"를 예측하는 모델을 학습하고, 예측 확률이
합성 비율 ``c = n_syn / (n_orig + n_syn)`` 에서 얼마나 벗어나는지 봅니다.

    pmse = mean over records i and classes k of (p_ik - c)^2

기존 구현을 그대로 유지해 ``predict_proba`` 두 열(원본일 확률, 합성일 확률)을 모두 평균합니다.
원본과 합성의 크기가 같으면(c = 0.5) Woo et al. 의 표준식 ``mean_i (p_i - c)^2`` 와 같습니다.

확률값 산출(:func:`compute_propensity_scores`)은 별도 함수로 분리되어 있으며
전처리의 ``pmse_probability`` 삭제 전략이 이 함수를 재사용합니다.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sdqe import config
from sdqe.preprocess.validation import Schema

logger = logging.getLogger("sdqe")

_MODEL_CLASSES = {
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
    "gradient_boosting": GradientBoostingClassifier,
}
_NAN_TOLERANT_MODELS = {"decision_tree"}


def build_model(model: str, model_params: dict[str, Any] | None, random_seed: int) -> Any:
    """pMSE 판별 모델을 만듭니다.

    Args:
        model: ``decision_tree`` | ``logistic_regression`` | ``gradient_boosting``.
        model_params: 모델 세부 파라미터 (sklearn 생성자 인자).
        random_seed: ``random_state`` 로 전달할 시드.

    Returns:
        학습 전 sklearn 분류기.

    Raises:
        ValueError: 지원하지 않는 모델이거나 잘못된 파라미터일 때.
    """
    if model not in _MODEL_CLASSES:
        raise ValueError(f"Unsupported --model '{model}'. Use one of {config.PMSE_MODELS}.")
    params = dict(model_params or {})
    params["random_state"] = random_seed
    try:
        return _MODEL_CLASSES[model](**params)
    except TypeError as exc:
        raise ValueError(f"Invalid --model_params for {model}: {exc}") from exc


@dataclass
class PropensityScores:
    """합성 여부 예측 결과.

    Attributes:
        proba: 예측 확률 행렬 (N, 2). 열 0 = 원본일 확률, 열 1 = 합성일 확률.
        predicted: 예측 라벨 (N,).
        labels: 실제 라벨 (N,). 0 = 원본, 1 = 합성. 앞 ``n_original`` 행이 원본.
        n_original: 원본 행 수.
        n_synthetic: 합성 행 수.
    """

    proba: np.ndarray
    predicted: np.ndarray
    labels: np.ndarray
    n_original: int
    n_synthetic: int

    @property
    def synthetic_probability(self) -> np.ndarray:
        """합성 레코드들의 "합성으로 판별될 확률" (n_synthetic,)."""
        return self.proba[self.n_original :, 1]


def design_matrix(original: pd.DataFrame, synthetic: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """원본과 합성을 합쳐 범주형을 더미 인코딩한 설계행렬 (기존 ``pd.get_dummies`` 방식)."""
    pooled = pd.concat(
        [original[list(schema.columns)], synthetic[list(schema.columns)]], ignore_index=True
    )
    return pd.get_dummies(pooled, columns=list(schema.categorical))


def compute_propensity_scores(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
    model: str = config.DEFAULT_PMSE_MODEL,
    model_params: dict[str, Any] | None = None,
    random_seed: int = config.DEFAULT_RANDOM_SEED,
) -> PropensityScores:
    """합성 여부 판별 모델을 학습하고 레코드별 확률값을 산출합니다.

    Args:
        original: 원본 데이터 (타입 정리 완료).
        synthetic: 합성 데이터 (타입 정리 완료).
        schema: 분석 스키마.
        model: 판별 모델 이름.
        model_params: 모델 세부 파라미터.
        random_seed: 모델 ``random_state``.

    Returns:
        :class:`PropensityScores`.

    References:
        [7] Woo, Reiter, Oganian and Karr (2009).
    """
    x = design_matrix(original, synthetic, schema)
    if model not in _NAN_TOLERANT_MODELS and x.isna().any().any():
        cols = [c for c in schema.numeric if x[c].isna().any()]
        raise ValueError(
            f"Model '{model}' does not support missing numeric values (columns: {cols}). "
            "Impute them or use --model decision_tree."
        )
    y = np.array([0] * len(original) + [1] * len(synthetic))
    clf = build_model(model, model_params, random_seed)
    features = x.to_numpy(dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(features, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        logger.debug(
            "pmse model '%s' did not fully converge; consider --model_params "
            "'{\"max_iter\": 1000}'.",
            model,
        )
    return PropensityScores(
        proba=clf.predict_proba(features),
        predicted=clf.predict(features),
        labels=y,
        n_original=len(original),
        n_synthetic=len(synthetic),
    )


def pmse_from_scores(scores: PropensityScores) -> dict[str, float]:
    """확률값으로 pMSE 와 판별 정확도를 계산합니다 (기존 식 유지)."""
    c = scores.n_synthetic / (scores.n_synthetic + scores.n_original)
    return {
        "value": float(np.mean((scores.proba - c) ** 2)),
        "accuracy": float(np.mean(scores.predicted == scores.labels)),
        "synthetic_share": c,
    }


def compute_pmse(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Schema,
    model: str = config.DEFAULT_PMSE_MODEL,
    model_params: dict[str, Any] | None = None,
    random_seed: int = config.DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """pMSE 를 계산합니다.

    Args:
        original: 원본 데이터.
        synthetic: 합성 데이터.
        schema: 분석 스키마.
        model: 판별 모델 이름.
        model_params: 모델 세부 파라미터.
        random_seed: 모델 ``random_state``.

    Returns:
        ``value``, ``accuracy``, ``synthetic_share`` 를 담은 딕셔너리.

    References:
        [7] Woo, Reiter, Oganian and Karr (2009).
    """
    scores = compute_propensity_scores(
        original, synthetic, schema, model, model_params, random_seed
    )
    return pmse_from_scores(scores)
