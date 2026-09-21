"""기본값 상수와 지표 목록.

지표 식별자, 임계값 방식 지원 범위, 판정 규칙은 이 파일 한 곳에서만 관리합니다.
식별자는 CLAUDE.md 용어 규칙 표와 동일한 문자열을 사용합니다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 지표 식별자 (고정)
# ---------------------------------------------------------------------------
UNIVARIATE_SIMILARITY = "univariate_similarity"
BIVARIATE_SIMILARITY = "bivariate_similarity"
PMSE = "pmse"
SINGLE_OUT_RISK = "single_out_risk"
INFERENCE_RISK = "inference_risk"
LINKAGE_RISK = "linkage_risk"

UTILITY_METRICS: tuple[str, ...] = (UNIVARIATE_SIMILARITY, BIVARIATE_SIMILARITY, PMSE)
PRIVACY_METRICS: tuple[str, ...] = (SINGLE_OUT_RISK, INFERENCE_RISK, LINKAGE_RISK)
ALL_METRICS: tuple[str, ...] = UTILITY_METRICS + PRIVACY_METRICS

METRIC_KOREAN_NAMES: dict[str, str] = {
    UNIVARIATE_SIMILARITY: "1차원 유사성",
    BIVARIATE_SIMILARITY: "2차원 유사성",
    PMSE: "pMSE",
    SINGLE_OUT_RISK: "구별위험도",
    INFERENCE_RISK: "추론위험도",
    LINKAGE_RISK: "연결위험도",
}

# ---------------------------------------------------------------------------
# 임계값 계산 방식 지원 범위 (명세 4.4절). 지원 여부는 이 테이블로만 판단합니다.
# ---------------------------------------------------------------------------
METHOD_SIMULATION = "simulation"
METHOD_DISTRIBUTION = "distribution"
THRESHOLD_METHODS: tuple[str, ...] = (METHOD_SIMULATION, METHOD_DISTRIBUTION)

THRESHOLD_METHOD_SUPPORT: dict[str, tuple[str, ...]] = {
    SINGLE_OUT_RISK: (METHOD_SIMULATION, METHOD_DISTRIBUTION),
    INFERENCE_RISK: (METHOD_SIMULATION, METHOD_DISTRIBUTION),
    BIVARIATE_SIMILARITY: (METHOD_SIMULATION,),
    PMSE: (METHOD_SIMULATION,),
}
THRESHOLD_METRICS: tuple[str, ...] = tuple(THRESHOLD_METHOD_SUPPORT)
DEFAULT_THRESHOLD_METHOD = METHOD_SIMULATION

THRESHOLD_SOURCE_SIMULATION = "simulation"
THRESHOLD_SOURCE_DISTRIBUTION = "distribution"
THRESHOLD_SOURCE_USER = "user_defined"

# 판정 규칙: 값이 임계값과 이 관계를 만족하면 통과입니다. (기존 코드의 비교 연산 유지)
PASS_RULES: dict[str, str] = {
    SINGLE_OUT_RISK: "<=",
    INFERENCE_RISK: "<",
    BIVARIATE_SIMILARITY: "<",
    PMSE: "<",
}

# 논문 범위 밖 확장 적용 지표에 기록하는 note (명세 4.5절)
EXTENDED_SCOPE_NOTE = "simulation framework extended beyond reference [2] scope"

# ---------------------------------------------------------------------------
# 공통 기본값
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "./results"
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_JOBS = -1
DEFAULT_N_SIMULATIONS = 100
MIN_RECOMMENDED_SIMULATIONS = 100
DEFAULT_QUANTILE = 0.95
ON_OFF = ("on", "off")

# 분포가정(이항비율 정규근사)이 부정확해지는 n*p*(1-p) 하한 (경험 규칙)
NORMAL_APPROX_MIN_NPQ = 10.0
# 이 행 수를 넘으면 시뮬레이션 대신 분포가정 방식을 안내합니다.
LARGE_DATA_ROWS = 20000

# ---------------------------------------------------------------------------
# 거리 측정 (common/distance.py)
# ---------------------------------------------------------------------------
DISTANCE_METRICS: tuple[str, ...] = ("gower", "cosine")
DEFAULT_DISTANCE_METRIC = "gower"
# 거리값을 이 자릿수로 반올림해 부동소수점 잡음(1e-12 미만)이 동률 판정을 좌우하지 않게 합니다.
DISTANCE_DECIMALS = 12
# 청크 하나가 쓰는 거리 행렬 메모리 상한 (bytes)
DISTANCE_CHUNK_BYTES = 64 * 1024 * 1024
# 시뮬레이션용 최근접 이웃 목록 길이. 50:50 분할에서 목록 안에 A 소속이 하나도 없을 확률은 0.5**K.
NEIGHBOR_LIST_SIZE = 64

# ---------------------------------------------------------------------------
# 안전성 지표
# ---------------------------------------------------------------------------
INFERENCE_REFERENCE_VALUE = 0.5
DEFAULT_LINKAGE_THRESHOLD = 0.7
LINKAGE_THRESHOLD_WARN_BELOW = 0.5
LINKAGE_MISSING_POLICIES: tuple[str, ...] = ("exclude", "error")
LINKAGE_EVALUATION_MODES: tuple[str, ...] = ("individual", "combined")
DEFAULT_LINKAGE_MISSING_POLICY = "exclude"
DEFAULT_LINKAGE_EVALUATION_MODE = "individual"

# ---------------------------------------------------------------------------
# 유용성 지표
# ---------------------------------------------------------------------------
# 기본값은 기존 Kdata 검증 코드와 같은 로지스틱 회귀입니다. 깊이 제한 없는 결정트리는 학습 데이터를
# 그대로 외워 pMSE 가 0.25 로 퇴화하므로 기본값으로 쓰지 않습니다 (2026-09-18 작성자 결정).
PMSE_MODELS: tuple[str, ...] = ("logistic_regression", "decision_tree", "gradient_boosting")
DEFAULT_PMSE_MODEL = "logistic_regression"
DEFAULT_ALPHA = 0.05

# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------
REDUCTION_METHODS: tuple[str, ...] = ("random", "pmse_probability", "inference_distance")
# inference_distance 전략의 목표값 = 임계값 * margin (기존 코드 EXCEEDED_THRESHOLD_MARGIN)
DEFAULT_TARGET_MARGIN = 0.95
PROCESSED_SYNTHETIC_FILENAME = "synthetic_processed.csv"

# 범주형 열에서 결측으로 의심되는 표기 (형식 검증 보고용)
MISSING_LIKE_TOKENS: tuple[str, ...] = (
    "",
    " ",
    "NA",
    "N/A",
    "na",
    "n/a",
    "null",
    "NULL",
    "None",
    "none",
    "nan",
    "NaN",
    "?",
    "-",
)

# 결과 캐시 스키마 버전. 캐시 내용 형식이 바뀌면 올립니다.
CACHE_VERSION = 1
