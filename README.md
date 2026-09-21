# sdqe

**S**ynthetic **D**ata **Q**uality **E**valuation — 합성(재현) 데이터의 유용성과 안전성을 정량 평가하는 CLI 도구

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Code style](https://img.shields.io/badge/lint-ruff-orange)](https://docs.astral.sh/ruff/)

원본 데이터와 합성 데이터를 비교해 6종의 품질 지표를 계산하고, 원본 데이터만으로 산출한 임계값과
대조해 통과 여부를 판정합니다. 「합성데이터 생성·활용 안내서」(개인정보보호위원회, 2024)와
안전성 지표 임계값 연구[[2]](#참고문헌)의 산식을 구현했습니다.

- **6종 지표** — 구별·추론·연결 위험도(안전성), 1차원·2차원 유사성·pMSE(유용성)
- **임계값 자동 산출** — 시뮬레이션(재표본) 또는 분포가정(정규근사) 방식
- **재현 가능** — 모든 난수는 `--random_seed`로 제어하고, 모든 산출물은 JSON으로 저장
- **한글 지원** — 변수명에 한글·공백·특수문자 사용 가능, 그림의 한글 폰트 자동 설정

```
설치 → sdqe report --original orig.csv --synthetic syn.csv → results/report.json
```

## 목차

[동작 개요](#동작-개요) · [설치](#설치) · [빠른 시작](#빠른-시작) · [평가 지표](#평가-지표) ·
[임계값 산출](#임계값-산출) · [CLI 레퍼런스](#cli-레퍼런스) · [설정 파일](#설정-파일) ·
[출력 구조](#출력-구조) · [문제 해결](#문제-해결) · [개발](#개발) · [참고문헌](#참고문헌)

## 동작 개요

평가는 세 단계로 구성되며 각 단계는 독립 실행할 수 있습니다.

```
① preprocess              ② threshold               ③ metric / utility / privacy
   원본 + 합성                원본만                     원본 + 전처리된 합성
   정합성 검증                판정 기준값 산출            지표값 계산 후 ②와 비교해 판정
   초과 레코드 삭제
                    └────────── report: ①②③ 일괄 실행 ──────────┘
```

단계별 산출물은 `--output_dir`(기본 `./results`) 아래에 저장되고 후속 단계가 자동으로 참조합니다.
임계값 JSON, 전처리된 합성 데이터, 거리 계산 캐시가 이 경로를 통해 연결됩니다.

**임계값을 원본만으로 산출하는 이유.** "원본 데이터는 그 자체로 가장 완벽한 재현 데이터"라는 가정
아래, 원본 내부에서 계산한 지표값은 *안전한 데이터에서 관측될 법한 값*의 표본이 됩니다. 이 표본
분포의 분위수(기본 95%)를 판정선으로 사용하므로 합성 데이터는 임계값 산출에 필요하지 않습니다.[[2]](#참고문헌)

## 설치

Python 3.9 이상이 필요합니다. 설치되어 있는지는 `python --version`으로 확인하십시오.

### 1. 내려받기

```bash
git clone https://github.com/mspark8788/sdqe.git
cd sdqe
```

### 2. 가상환경 만들기 (권장)

프로젝트 전용 공간을 만들어 시스템 파이썬을 건드리지 않습니다.

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

프롬프트 앞에 `(.venv)`가 붙으면 활성화된 것입니다. 새 터미널을 열 때마다 다시 활성화해야 합니다.

> Windows에서 `실행할 수 없습니다` 오류가 나면 실행 정책 때문입니다.
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` 를 먼저 실행하십시오.

### 3. 설치

```bash
pip install .
```

`sdqe` 명령이 만들어집니다. 용도에 따라 아래를 대신 쓸 수 있습니다.

| 명령 | 언제 쓰나 | 추가되는 것 |
|---|---|---|
| `pip install .` | 평가만 수행할 때 | — |
| `pip install -e ".[dev]"` | 코드를 수정할 때 | 소스 수정이 즉시 반영(`-e`), pytest·ruff·statsmodels |
| `pip install ".[gpu]"` | cosine 거리를 GPU로 계산할 때 | torch |

`[dev]`, `[gpu]`는 선택 묶음이며 따옴표가 필요합니다(셸이 `[ ]`를 해석하지 않도록).

### 4. 확인

```bash
sdqe --help
```

`python -m sdqe --help`로도 동일하게 동작합니다. 실제로 평가가 도는지 보려면
아래 [빠른 시작](#빠른-시작)의 명령을 그대로 실행하십시오. 저장소에 표본 데이터가 들어 있어
별도 준비 없이 결과까지 확인할 수 있습니다.

### clone 없이 설치하기

코드를 수정할 계획이 없다면 한 줄로 끝낼 수도 있습니다.

```bash
pip install git+https://github.com/mspark8788/sdqe.git
```

이 경우 저장소 파일(표본 데이터, 설정 예시)은 받아지지 않고 `sdqe` 명령만 설치됩니다.

## 빠른 시작

전처리부터 전 지표 평가까지 한 번에 실행합니다. 저장소에 포함된 [test_data/](test_data/) 샘플로

바로 확인할 수 있습니다.

```bash
sdqe report --original test_data/adult_origin.csv \
            --synthetic test_data/adult_synth.csv \
            --output_dir ./results \
            --reduction_method inference_distance \
            --quasi_identifiers gender race marital_status \
            --sensitive_attributes income
```

```
univariate_similarity  value=n/a       judgement=n/a
bivariate_similarity   value=0.039216  judgement=PASSED
pmse                   value=0.011595  judgement=PASSED
single_out_risk        value=0.000000  judgement=PASSED
inference_risk         value=0.450102  judgement=PASSED
linkage_risk           value=0.716016  judgement=NOT PASSED
```

동일한 내용이 `results/report.json`에 기록되고, 지표별 상세는 `results/metrics/<지표>.json`,
1차원 분포 그림은 `results/figures/`에 저장됩니다.

> **Note**
> `linkage_risk`는 준식별자(`--quasi_identifiers`)와 민감정보(`--sensitive_attributes`)를
> 지정해야 계산합니다. 지정하지 않으면 경고를 남기고 해당 지표만 건너뜁니다.

단계별 실행은 [CLI 레퍼런스](#cli-레퍼런스)를 참고하십시오.

## 평가 지표

| 구분 | 식별자 | 명칭 | 값의 의미 | 통과 조건 |
|---|---|---|---|---|
| 안전성 | `single_out_risk` | 구별위험도 | 클수록 위험 | `값 ≤ 임계값` |
| 안전성 | `inference_risk` | 추론위험도 | 이론 기준값 0.5 | `값 < 임계값` |
| 안전성 | `linkage_risk` | 연결위험도 (CAP) | 레코드별 0~1 | **레코드 단위**: `CAP > 임계값`인 레코드 0건 |
| 유용성 | `univariate_similarity` | 1차원 유사성 | 변수별 분포 검정 결과 | 판정 없음 |
| 유용성 | `bivariate_similarity` | 2차원 유사성 | 작을수록 변수 간 관계가 유사 | `값 < 임계값` |
| 유용성 | `pmse` | pMSE | 작을수록 원본과 구별 곤란 | `값 < 임계값` |

### single_out_risk — 구별위험도

합성 레코드와 **완전히 일치하는** 원본 레코드의 비율입니다.

```
식 (2) 기본:  p = (1/n) Σ I(합성 레코드 i 가 원본에 존재) / d_i
식 (1) 선택:  p = (1/n) Σ I(합성 레코드 i 가 원본에 존재)
```

`d_i`는 해당 레코드와 동일한 원본 레코드의 개수입니다. 유일한 레코드일수록 위험이 크고 원본 내
중복이 많을수록 위험이 낮아지는 성질을 반영합니다. `--duplicate_adjustment off`로 식 (1)을
선택할 수 있습니다. 안내서[[1]](#참고문헌)는 식 (1)을 기본, 식 (2)를 변형으로 제시하나
본 구현은 논문[[2]](#참고문헌)을 따라 식 (2)를 기본값으로 둡니다.

### inference_risk — 추론위험도

완전 일치가 없더라도 거리가 충분히 가까우면 속성 추론이 가능하다는 관점의 지표입니다.

```
합성 레코드 i 에 대해
    d_i     = 최근접 원본 레코드 r_k 까지의 거리       (nearest_distance)
    d_or,i  = r_k 의 최근접 이웃(다른 원본)까지의 거리  (reference_distance)
    d_i < d_or,i 이면 지시함수 1

p = (지시함수 1의 개수) / (전체 - 동률 레코드 수)
```

- 동률(`d_i == d_or,i`) 레코드는 분모에서 제외하며[[1, 68쪽]](#참고문헌), 제외 건수를 `n_ties`로 보고합니다.
- **이론 기준값은 0.5**입니다. 0.5를 크게 상회하면 추론 위험이, 크게 하회하면 유용성 저하가 의심됩니다.
  실제 판정은 0.5가 아니라 산출된 임계값으로 수행합니다.

거리 함수는 `--distance_metric`으로 선택합니다.

| 값 | 정의 |
|---|---|
| `gower` (기본) | 범주형은 불일치 여부(0/1), 수치형은 원본 min~max 범위로 정규화한 절대차. 변수 수로 평균 |
| `cosine` | 범주형 일치율과 표준화된 수치형의 코사인 유사도를 변수 수로 가중평균한 뒤 `1 - 유사도` |

스케일 기준(범위·평균·표준편차)은 **원본에서만** 산출해 합성 데이터에 동일하게 적용합니다.
최근접 후보가 동률이면 원본의 행 번호가 가장 작은 레코드를 선택합니다.

### linkage_risk — 연결위험도 (CAP)

공격자가 준식별자 K를 알고 있을 때 민감정보 T를 정확히 추정할 확률이며, **원본 레코드별로** 계산합니다.

```
CAP_j = (합성에서 K, T 가 모두 일치하는 레코드 수) / (합성에서 K 가 일치하는 레코드 수)
```

- K가 일치하는 합성 레코드가 없으면 CAP을 0으로 두고 평균에 포함하며, 건수를 `unmatched_count`로 보고합니다.
- 임계값은 산출하지 않고 `--linkage_threshold`로 지정합니다(기본 0.7, `0.0 ≤ t < 1.0`).
- **판정은 레코드 단위입니다.** `CAP > 임계값`인 레코드가 한 건이라도 있으면 불통과이므로,
  참고값인 평균(`value`)이 임계값보다 낮아도 불통과가 나올 수 있습니다. `exceed_count`,
  `exceed_ratio`를 함께 확인하십시오.
- 준식별자·민감정보는 범주형이어야 합니다. 수치형은 사전 구간화가 필요합니다.
- 값은 문자열로 비교하므로 `01`과 `1`, `1`과 `1.0`은 서로 다른 범주로 처리됩니다.

### univariate_similarity — 1차원 유사성

변수별로 원본과 합성의 주변 분포를 비교합니다. 임계값 없이 검정 결과를 그대로 보고합니다.

- 수치형: Kolmogorov-Smirnov 검정 + ECDF 그림
- 범주형: 카이제곱 검정 + 범주 비율 막대그림
- 변수별 그림과 전체 변수를 한 장에 담은 요약 그림을 저장합니다.

### bivariate_similarity — 2차원 유사성

변수 쌍의 연관성 행렬을 원본과 합성에서 각각 구성하고, 두 행렬 차이의 표준편차를 지표값으로 사용합니다.

| 변수 쌍 | 연관성 계수 |
|---|---|
| 수치형 – 수치형 | Pearson 상관계수 |
| 수치형 – 범주형 | 일원 분산분석 OLS의 결정계수 |
| 범주형 – 범주형 | Cramér's V |

모든 값이 동일한 상수 열은 연관성 계산이 불가능하므로 `--drop_constant_columns on`(기본)에서
제외하고, 제외한 열 목록을 결과에 기록합니다.

### pmse — Propensity Mean Squared Error

원본과 합성을 결합한 뒤 "합성 여부"를 예측하는 판별 모델을 학습하고, 예측 확률이 합성 비율에서
이탈한 정도를 측정합니다. 구별이 어려울수록(유용성이 높을수록) 값이 작아집니다.

- `--model`: `logistic_regression`(기본), `decision_tree`, `gradient_boosting`
- `--model_params`: 모델 세부 설정. 예) `--model_params '{"max_depth": 5}'`
- 깊이를 제한하지 않은 결정트리는 학습 데이터를 암기해 값이 최댓값 0.25에 고정됩니다.
  결정트리 사용 시 `max_depth` 등의 제약을 함께 지정하십시오.

## 임계값 산출

### 지표별 지원 방식

| 지표 | `--method simulation` | `--method distribution` | 비고 |
|---|---|---|---|
| `single_out_risk` | 지원 | 지원 | `--strict`로 0 지정 가능 |
| `inference_risk` | 지원 | 지원 | |
| `bivariate_similarity` | 지원 | 미지원 | 논문 범위 밖 확장 |
| `pmse` | 지원 | 미지원 | 논문 범위 밖 확장 |
| `linkage_risk` | — | — | 산출하지 않음. `--linkage_threshold`로 지정 |
| `univariate_similarity` | — | — | 임계값 없음 |

미지원 조합은 다음과 같이 거부합니다.

```
ERROR --method distribution is not supported for metric 'pmse'.
      Supported methods: simulation
```

### 시뮬레이션 방식 (기본)

1. 원본을 50:50으로 무작위 분할해 한쪽을 원본(A), 다른 쪽을 재현(B)으로 간주합니다.
2. A, B로 지표를 계산합니다.
3. 1~2를 `--n_simulations`회(기본 100) 반복하고, 값들의 `--quantile` 분위수(기본 0.95)를 임계값으로 씁니다.

- **분위수의 의미**: 0.95는 실제로 안전한 합성 데이터를 5% 확률로 불통과 판정한다는 뜻입니다.
  기준을 강화하려면 0.90, 완화하려면 0.99를 사용합니다.
- 반복 횟수가 100 미만이면 경고합니다. 반복별 시드와 값은 `threshold/<지표>_simulations.json`에 전량 기록됩니다.
- **`single_out_risk` 크기 보정**: 시뮬레이션은 n/2 크기끼리 비교하므로 크기 n의 합성 데이터와
  대조하려면 `p* = 1 - (1 - p)²` 보정이 필요합니다(`--size_correction on`, 기본). `p`와 `p*`를 모두 기록합니다.

### 분포가정 방식

이등분과 반복 없이 원본 전체에서 비율 p를 한 번만 구하고, 이항비율의 정규근사
`p̂ ~ N(p, p(1-p)/n)`의 분위수를 임계값으로 사용합니다. 대용량 데이터에서 계산 시간이 크게 줄어듭니다.

- `single_out_risk`: 각 원본 레코드를 재현 레코드로 간주하고 나머지 원본에 동일 레코드가 있는지 확인합니다.
- `inference_risk`: 원본 레코드 r₁의 최근접 r_k까지 거리 d₁과, r₁을 제외한 r_k의 최근접 거리 d₂를 비교합니다.
- 이등분하지 않으므로 크기 보정(`p*`)은 적용하지 않습니다.
- `n·p·(1-p) < 10`이면 근사 정확도가 떨어지므로 경고하고 시뮬레이션 사용을 권고합니다.

### 사용자 지정

| 인자 | 동작 |
|---|---|
| `--strict` | `single_out_risk` 임계값을 계산 없이 0으로 설정 (완전 일치 레코드 불허) |
| `--manual_threshold 0.3` | 임계값을 직접 지정 |

두 경우 모두 결과에 `"threshold_source": "user_defined"`로 기록됩니다.

## CLI 레퍼런스

### 공통 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--original`, `--synthetic` | (필수) | csv / parquet / xlsx. CSV는 utf-8-sig로 읽고 실패 시 cp949로 재시도 |
| `--output_dir` | `./results` | 결과 저장 위치 |
| `--categorical_columns` | dtype 추론 | 범주형으로 취급할 열. 숫자 코드형 범주 변수는 반드시 지정 |
| `--exclude_columns` | 없음 | 모든 계산에서 제외할 열 |
| `--random_seed` | `42` | 분할·표본추출·모델 학습의 모든 난수 |
| `--n_jobs` | `-1` | 거리 계산 병렬 스레드 수 |
| `--verbose` | off | 상세 로그 |

전체 인자는 `sdqe <서브커맨드> --help`로 확인할 수 있습니다.

### preprocess

변수 이름·순서·타입·형식을 검증하고, 합성이 원본보다 크면 초과 레코드를 삭제합니다.

```bash
sdqe preprocess --original orig.csv --synthetic syn.csv --output_dir ./results \
                --reduction_method inference_distance
```

- 변수명이나 타입(범주형/수치형)이 다르면 자동 보정하지 않고 중단하며, 불일치 내역을 출력합니다.
- 열 순서가 다르면 원본 순서로 정렬하고 그 사실을 기록합니다.
- `--reduction_method`를 생략하면 삭제 없이 검증만 수행합니다.
- `--remove_exact_matches on`은 원본과 완전히 동일한 합성 레코드를 먼저 제거합니다(기본 off).
  구별위험도를 0으로 만들지만 그만큼 유용성이 감소합니다.

| 삭제 전략 | 기준 |
|---|---|
| `random` | 무작위 |
| `pmse_probability` | 합성으로 판별될 확률이 높은(유용성이 낮은) 레코드부터 |
| `inference_distance` | 추론위험도 지시함수값 기준. 목표치(임계값 × `--target_margin`, 기본 0.95) 이내가 되도록 1과 0을 비율에 맞춰 삭제 |

`inference_distance`는 추론위험도 임계값이 필요합니다. `threshold`를 먼저 실행하거나
`--inference_threshold`로 직접 지정하십시오. 이때 계산한 거리는 캐시에 저장되어 후속 지표 계산에서 재사용됩니다.

산출물: `preprocess/validation.json`, `preprocess/reduction.json`,
`preprocess/synthetic_processed.csv`(원자료 표기 유지, 열 순서만 원본 기준).

### threshold

원본 데이터만 사용합니다.

```bash
sdqe threshold --metric inference_risk --original orig.csv --output_dir ./results \
               --method simulation --n_simulations 100 --quantile 0.95 --distance_metric gower

sdqe threshold --metric single_out_risk --original orig.csv --output_dir ./results --method distribution
sdqe threshold --metric single_out_risk --original orig.csv --output_dir ./results --strict
sdqe threshold --metric pmse --original orig.csv --output_dir ./results
sdqe threshold --metric bivariate_similarity --original orig.csv --output_dir ./results
```

### metric / utility / privacy

전처리된 합성 데이터를 입력으로 사용합니다.

```bash
# 단일 지표
sdqe metric --name inference_risk --original orig.csv \
            --synthetic results/preprocess/synthetic_processed.csv --output_dir ./results

# 유용성 일괄
sdqe utility --original orig.csv --synthetic results/preprocess/synthetic_processed.csv \
             --output_dir ./results

# 안전성 일괄
sdqe privacy --original orig.csv --synthetic results/preprocess/synthetic_processed.csv \
             --output_dir ./results \
             --quasi_identifiers gender race --sensitive_attributes income
```

`<output_dir>/threshold/<지표>.json`이 존재하면 자동으로 판정하고, 없으면 값만 계산한 뒤 판정을 비워 둡니다.
`--metrics`로 일부만 선택하거나 `--skip_metrics`로 일부를 제외할 수 있습니다.

### report

임계값 → 전처리 → 지표 순으로 일괄 실행합니다. 임계값은 원본만 필요하고 `inference_distance`
전처리가 추론위험도 임계값을 사용하므로 가장 먼저 계산합니다.

## 설정 파일

`utility`, `privacy`, `report`는 `--config config.yaml`로 인자를 받을 수 있습니다. 키 이름은 CLI
인자와 동일하며 **CLI 인자가 설정 파일보다 우선**합니다. 정의되지 않은 키는 에러로 처리합니다.

```yaml
original: test_data/adult_origin.csv
synthetic: test_data/adult_synth.csv
output_dir: ./results
categorical_columns: [education_num]   # 숫자로 코딩된 범주형이 있을 때
exclude_columns: [fnlwgt]
reduction_method: inference_distance
distance_metric: gower
method: distribution        # 안전성 임계값 방식 (유용성은 시뮬레이션으로 자동 전환)
n_simulations: 100
quantile: 0.95
model: logistic_regression
quasi_identifiers: [gender, race, marital_status]
sensitive_attributes: [income]
linkage_threshold: 0.7
random_seed: 42
```

```bash
sdqe report --config config.yaml
```

## 출력 구조

```
results/
├── preprocess/   validation.json, reduction.json, synthetic_processed.csv
├── threshold/    <지표>.json, <지표>_simulations.json
├── metrics/      <지표>.json, linkage_risk_records.json
├── figures/      univariate_similarity_<변수명>.png, univariate_similarity_summary.png
├── cache/        <해시>.json   (추론위험도 거리 계산 결과)
└── report.json
```

### 지표 JSON 공통 필드

| 필드 | 설명 |
|---|---|
| `value` | 지표값 |
| `threshold`, `threshold_source` | 판정에 사용한 임계값과 산출 방식(`simulation` / `distribution` / `user_defined`) |
| `pass_rule` | 통과 조건 (예: `value < threshold`) |
| `passed` | 판정 결과. 임계값이 없으면 `null` |
| `parameters` | 계산에 사용한 인자 (거리 방식, 모델, 시드 등) |
| `input_info` | 원본/합성 행 수, 변수 수 |
| `excluded_columns` | 계산에서 제외한 열 |
| `computed_at`, `elapsed_seconds` | 실행 시각과 소요 시간 |
| `details` | 지표별 상세 |

### 지표별 주요 필드

| 지표 | 필드 | 설명 |
|---|---|---|
| `inference_risk` | `reference_value` | 이론 기준값 0.5 |
| | `details.n_ties`, `details.n_evaluated` | 동률로 제외한 레코드 수와 실제 분모 |
| | `cache_hit` | 캐시된 거리 계산 재사용 여부 |
| `single_out_risk` | `details.n_matched_records` | 원본과 완전히 일치하는 합성 레코드 수 |
| `linkage_risk` | `exceed_count`, `exceed_ratio` | 임계값 초과 레코드 수와 비율 (판정 근거) |
| | `details.records_file` | 원본 레코드별 CAP 전체 |
| `bivariate_similarity` | `details.pairs` | 변수 쌍별 연관성 계수와 차이 |
| `pmse` | `details.accuracy` | 판별 모델 정확도. 1.0이면 과적합 의심 |
| `univariate_similarity` | `details.variables` | 변수별 검정 통계량, p-value, 그림 경로 |

### 임계값 JSON

- 시뮬레이션: `simulation.mean`, `std`, `quantile_value`, 원시값 파일 경로
- 분포가정: `distribution.mean`(p), `variance`, `n`, `npq`
- `single_out_risk`: `single_out_risk.p_hat`, `p_star`, `size_correction`

### 판정 해석

- 통과 조건은 지표마다 다릅니다. [평가 지표 표](#평가-지표)를 참조하십시오.
- **연결위험도는 평균과 판정이 분리되어 있습니다.** 평균 CAP이 임계값 근처이거나 낮더라도
  임계값을 초과하는 레코드가 한 건이라도 있으면 불통과입니다. 로그에 함께 표시됩니다.
  ```
  linkage_risk = 0.716016 | threshold 0.700000 (user_defined) | 256 of 500 records (51.2%) exceed it NOT PASSED
  ```
- 추론위험도가 임계값보다 낮아 통과하더라도 0.5를 크게 하회하면 유용성 저하 신호입니다.
- 1차원 유사성은 판정이 없습니다. p-value가 작으면 해당 변수의 분포가 다르다는 의미이므로 그림과 함께 확인하십시오.

## 문제 해결

<details>
<summary><code>distance_metric mismatch</code> 에러가 발생합니다</summary>

임계값을 `cosine`으로 산출하고 지표를 `gower`로 계산하면 비교가 성립하지 않으므로 중단합니다.
둘 중 하나를 동일한 `--distance_metric`으로 다시 계산하십시오.
</details>

<details>
<summary>판정이 비어 있습니다 (<code>passed: null</code>)</summary>

해당 지표의 임계값 파일이 없습니다. `sdqe threshold --metric <지표>`를 먼저 실행하거나
`--threshold_file`로 경로를 지정하십시오.
</details>

<details>
<summary>합성 데이터가 원본보다 큽니다</summary>

`--reduction_method`를 지정하면 원본 크기에 맞춰 삭제합니다. 지정하지 않으면 경고만 남기고 그대로
계산합니다. 삭제 기준과 삭제된 행 번호는 `preprocess/reduction.json`에 기록됩니다.
</details>

<details>
<summary>계산이 오래 걸립니다</summary>

추론위험도의 거리 계산이 전체 수행 시간을 좌우합니다.

- 대용량 데이터는 안전성 임계값을 `--method distribution`으로 산출하십시오(단일 계산).
- `--n_jobs`로 스레드 수를 늘리십시오.
- `cosine`은 `--use_gpu`로 GPU를 사용할 수 있습니다. torch나 CUDA가 없으면 경고 후 CPU로 진행합니다.
- 입력이 동일하면 거리 계산 결과를 캐시에서 재사용하며, 입력 파일을 수정하면 캐시는 자동 무효화됩니다.
</details>

<details>
<summary>그림의 한글이 깨집니다</summary>

맑은 고딕, 나눔고딕 등 한글 폰트를 찾지 못한 경우입니다. 경고를 남기고 대체 폰트로 진행하므로
폰트를 설치하면 해결됩니다.
</details>

<details>
<summary>상수 열 때문에 값이 NaN으로 나옵니다</summary>

`bivariate_similarity`에서 발생합니다. 기본값 `--drop_constant_columns on`이면 자동으로 제외하고
제외 목록을 결과에 기록합니다.
</details>

<details>
<summary>숫자로 코딩된 범주형 변수가 수치형으로 인식됩니다</summary>

`--categorical_columns 컬럼1 컬럼2`로 명시하십시오. 지정하지 않으면 dtype으로 추론합니다.
</details>

## 개발

```bash
pip install -e ".[dev]"
pytest -q                   # 테스트
ruff check src tests        # 린트
ruff format src tests       # 포맷
```

### 테스트 데이터

[test_data/](test_data/)의 `adult_origin.csv`(500행), `adult_synth.csv`(700행)는 UCI Adult
데이터셋에서 고정 시드(42)로 추출한 표본입니다. README 예시와 통합 테스트가 이 파일을 사용합니다.
합성본은 원본보다 크므로 `preprocess`의 축소 경로까지 함께 확인할 수 있습니다.

## 참고문헌

1. 개인정보보호위원회, 「합성데이터 생성·활용 안내서」, 2024. 12.
   — 안전성 지표 정의(구별·연결·추론 위험도)와 산출 절차
2. 박민수, 정미령, 김승환, 박헌진, "재현 데이터 안전성 평가 지표의 임계값 연구",
   *Journal of Korean Institute of Intelligent Systems*, 34(6), 2024.
   — 안전성 지표 수식(식 1~4)과 임계값 산출(시뮬레이션, 분포가정, 식 5)
3. 안성빈 외, "유용성과 노출 위험성 지표를 이용한 재현자료 기법 비교 연구",
   *응용통계연구*, 36(2), 141–166, 2023. https://doi.org/10.5351/KJAS.2023.36.2.141
4. Article 29 Data Protection Working Party, "Opinion 05/2014 on Anonymisation Techniques", 2014.
   — Singling out / Linkability / Inference
5. Taub, J. and Elliot, M., "The Synthetic Data Challenge", *UNECE Conference of European Statisticians*, 2019. — CAP
6. Gower, J. C., "A General Coefficient of Similarity and Some of Its Properties",
   *Biometrics*, 27(4), 857–871, 1971. — Gower 거리
7. Woo, M.-J., Reiter, J. P., Oganian, A. and Karr, A. F., "Global Measures of Data Utility for Microdata
   Masked for Disclosure Limitation", *Journal of Privacy and Confidentiality*, 1(1), 2009. — pMSE
8. Becker, B. and Kohavi, R., "Adult", *UCI Machine Learning Repository*, 1996. https://doi.org/10.24432/C5XW20
   — `test_data/`의 표본 데이터 출처 (CC BY 4.0)
