"""데이터 로드, JSON 저장, 결과 캐시 조회.

캐시 해시 생성 함수(:func:`make_cache_key`)는 이 파일 한 곳에만 둡니다.
전처리(`inference_distance` 전략)와 지표 계산(`inference_risk`)이 같은 함수를 호출합니다.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdqe import config

logger = logging.getLogger("sdqe")

_CSV_ENCODINGS = ("utf-8-sig", "cp949")


# ---------------------------------------------------------------------------
# 데이터 로드 / 저장
# ---------------------------------------------------------------------------
def read_table(path: str | Path) -> pd.DataFrame:
    """CSV, Parquet, Excel 파일을 DataFrame으로 읽습니다.

    CSV는 ``utf-8-sig`` 로 먼저 읽고 실패하면 ``cp949`` 로 다시 시도합니다.

    Args:
        path: 입력 파일 경로.

    Returns:
        읽어 들인 DataFrame.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        ValueError: 지원하지 않는 확장자이거나 인코딩을 판별하지 못했을 때.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix not in (".csv", ".txt"):
        raise ValueError(f"Unsupported file extension '{suffix}': {path}")
    last_error: Exception | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode {path} with encodings {_CSV_ENCODINGS}: {last_error}")


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """DataFrame을 ``utf-8-sig`` CSV로 저장합니다.

    Args:
        df: 저장할 DataFrame.
        path: 저장 경로. 상위 디렉터리가 없으면 만듭니다.

    Returns:
        저장한 파일 경로.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def to_jsonable(obj: Any) -> Any:
    """NumPy/pandas 값을 JSON 직렬화 가능한 파이썬 기본형으로 바꿉니다.

    NaN 과 무한대는 ``None`` (JSON null) 으로 기록합니다.

    Args:
        obj: 변환할 객체.

    Returns:
        JSON으로 저장할 수 있는 객체.
    """
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """딕셔너리를 UTF-8 JSON 파일로 저장합니다.

    Args:
        data: 저장할 딕셔너리.
        path: 저장 경로. 상위 디렉터리가 없으면 만듭니다.

    Returns:
        저장한 파일 경로.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    """JSON 파일을 읽습니다.

    Args:
        path: JSON 파일 경로.

    Returns:
        읽어 들인 딕셔너리.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def now_iso() -> str:
    """현재 시각을 로컬 시간대가 붙은 ISO 8601 문자열로 반환합니다."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def stage_dir(output_dir: str | Path, stage: str) -> Path:
    """``output_dir/<stage>`` 디렉터리를 만들고 경로를 반환합니다.

    Args:
        output_dir: 결과 최상위 디렉터리.
        stage: ``preprocess``, ``threshold``, ``metrics``, ``figures``, ``cache`` 중 하나.

    Returns:
        생성된 디렉터리 경로.
    """
    path = Path(output_dir) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 결과 캐시 (명세 3.3절)
# ---------------------------------------------------------------------------
def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """파일 경로와 수정시각으로 된 지문을 만듭니다.

    Args:
        path: 대상 파일 경로.

    Returns:
        ``{"path": 절대경로, "mtime_ns": 수정시각}``.
    """
    resolved = Path(path).resolve()
    return {"path": resolved.as_posix(), "mtime_ns": resolved.stat().st_mtime_ns}


def make_cache_key(
    original_path: str | Path,
    synthetic_path: str | Path,
    distance_metric: str,
    params: Mapping[str, Any] | None = None,
) -> str:
    """입력 해시를 만듭니다.

    해시는 (원본 경로+수정시각, 합성 경로+수정시각, ``distance_metric``, 관련 파라미터)로
    구성합니다. 전처리와 지표 계산이 이 함수 하나만 사용합니다.

    Args:
        original_path: 원본 데이터 경로.
        synthetic_path: 합성 데이터 경로.
        distance_metric: 거리 측정 방식.
        params: 결과에 영향을 주는 추가 파라미터(사용 열, 범주형 열 등).

    Returns:
        SHA-256 해시 앞 24자리 문자열.
    """
    payload = {
        "cache_version": config.CACHE_VERSION,
        "original": file_fingerprint(original_path),
        "synthetic": file_fingerprint(synthetic_path),
        "distance_metric": distance_metric,
        "params": to_jsonable(dict(params or {})),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def cache_file(output_dir: str | Path, key: str) -> Path:
    """캐시 키에 해당하는 파일 경로 ``output_dir/cache/<key>.json`` 을 반환합니다."""
    return Path(output_dir) / "cache" / f"{key}.json"


def load_cache(output_dir: str | Path, key: str) -> dict[str, Any] | None:
    """캐시를 조회합니다.

    Args:
        output_dir: 결과 최상위 디렉터리.
        key: :func:`make_cache_key` 로 만든 키.

    Returns:
        캐시 내용. 없거나 읽을 수 없으면 ``None``.
    """
    path = cache_file(output_dir, key)
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None


def save_cache(output_dir: str | Path, key: str, payload: Mapping[str, Any]) -> Path:
    """캐시를 저장합니다.

    Args:
        output_dir: 결과 최상위 디렉터리.
        key: :func:`make_cache_key` 로 만든 키.
        payload: 저장할 내용.

    Returns:
        저장한 파일 경로.
    """
    data = {"cache_key": key, "created_at": now_iso(), **payload}
    return save_json(data, cache_file(output_dir, key))


def read_table_as_strings(path: str | Path, na_values: list[str] | None = None) -> pd.DataFrame:
    """파일의 값을 형 변환 없이 문자열 그대로 읽습니다.

    ``linkage_risk`` 는 값을 문자열로 비교하는 기존 방식을 유지하기 위해,
    전처리는 결과 파일에 원자료 표기를 그대로 보존하기 위해 사용합니다.

    Args:
        path: 입력 파일 경로.
        na_values: 결측으로 볼 표기. ``None`` 이면 어떤 값도 결측으로 바꾸지 않습니다.

    Returns:
        모든 열이 문자열(``string`` dtype)인 DataFrame.
    """
    path = Path(path)
    if path.suffix.lower() not in (".csv", ".txt"):
        return read_table(path).astype("string")
    last_error: Exception | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            return pd.read_csv(
                path,
                dtype="string",
                encoding=encoding,
                keep_default_na=False,
                na_values=na_values or [],
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode {path} with encodings {_CSV_ENCODINGS}: {last_error}")
