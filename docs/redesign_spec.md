# 재설계 상세 명세

이 문서는 `sdqe`(합성데이터 품질평가 패키지) 재설계의 구현 요구사항을 담습니다.
세션마다 항상 필요한 규칙과 **용어 명명 규칙**은 프로젝트 루트 `CLAUDE.md`에 있고,
이 문서는 **작업할 섹션을 지목해서 참조**하는 용도입니다.

- 작성 기준일: 2026-09 (3차 개정, 참고문헌 [2] 논문 반영)
- 안전성 지표와 임계값 계산의 근거는 참고문헌 [2] 박민수 외(2024)입니다.
- 관련 규칙 파일: `.claude/rules/` 하위 파일들

---

## 0. 목차

| 장 | 내용 |
|---|---|
| 1 | 전체 설계 원칙 및 지표 정의 |
| 2 | CLI 실행 구조 |
| 3 | 전처리 |
| 4 | 임계값 계산 |
| 5 | 유용성 지표 |
| 6 | 안전성 지표 |
| 7 | 결과 파일 규격 |
| 8 | 검증용 참조값 |
| 9 | 미확정 사항 (확인 필요) |
| 10 | 참고문헌 |

---

## 1. 전체 설계 원칙 및 지표 정의

1. 전반적인 코드 구조를 논리적이고 가독성 높게 재설계합니다.
2. 계산 방식을 철저히 지켜 **재현 가능한 결과**를 보장합니다.
3. `--` 형태의 argparse 인자로 지표를 계산할 수 있게 합니다.
4. 수학적 계산 코드이므로 **속도 최적화**를 함께 고려합니다.
5. 현재 코드의 `log.txt` 생성과 pickle 저장 기능은 제거합니다.
6. 임계값 계산과 지표 계산은 **각각 따로 실행**할 수 있어야 합니다.
7. 모든 계산 결과는 JSON 등으로 기술적으로 관리합니다.
8. 큰 흐름은 전처리 → 임계값 계산 → 지표 계산입니다.

### 1.1 지표 식별자 (고정)

`CLAUDE.md`의 용어 규칙과 동일합니다. 이 문서 전체에서 아래 식별자만 사용합니다.

| 구분 | 식별자 | 한글 명칭 | 논문 대응 | 임계값 계산 대상 |
|---|---|---|---|---|
| 유용성 | `univariate_similarity` | 1차원 유사성 | - | 아니오 |
| 유용성 | `bivariate_similarity` | 2차원 유사성 | - | 예 |
| 유용성 | `pmse` | pMSE | - | 예 |
| 안전성 | `single_out_risk` | 구별위험도 | Single Out Risk Metric, 식 (2) | 예 |
| 안전성 | `inference_risk` | 추론위험도 | Inference Risk Metric, 식 (4) | 예 |
| 안전성 | `linkage_risk` | 연결위험도 | CAP, 식 (3) | 아니오 (사용자 지정) |

### 1.2 안전성 지표의 정의 (참고문헌 [2] 2장)

논문은 Article 29 Data Protection Working Party가 정의한 세 가지 재식별 위험
(Singling out, Linkability, Inference)에 각각 대응하는 지표를 제시합니다.

**`single_out_risk` (구별위험도, 식 (2))**

재현 데이터의 레코드와 동일한 원본 레코드를 찾을 수 있는 비율입니다.
단순 일치 비율인 식 (1)이 아니라, **원본 데이터 내 중복 레코드 수를 고려한
식 (2)를 기본 구현으로 삼습니다.**

```
단순형(식 1)   p_hat = (1/n) * sum_i  I(r_i == s_i)
중복 보정형(식 2) p_hat = (1/n) * sum_i  I(r_i == s_i) / d_i
```

`d_i` 는 원본 데이터 내에서 해당 레코드와 동일한 레코드의 개수입니다.
유일한 레코드는 노출 위험이 크고, 중복이 많으면 위험이 낮아지는 성질을 반영합니다.
범주형 변수 위주의 데이터는 조합이 유한하여 원본 내 중복이 많으므로 식 (2)가 필요합니다.

`--duplicate_adjustment {on, off}` 로 전환할 수 있게 하되 기본값은 `on`(식 2)입니다.

**`linkage_risk` (연결위험도, CAP, 식 (3))**

공격자가 준식별자(Quasi-Identifier, K)를 알고 있을 때 재현 데이터에서
민감정보(Sensitive Attribute, T)를 맞출 확률입니다.

```
CAP_j = (준식별자와 민감정보가 모두 일치하는 원본 레코드 수)
        / (준식별자가 일치하는 원본 레코드 수)
```

재현 데이터 레코드마다 하나씩, 총 n개가 계산됩니다.
따라서 준식별자 열과 민감정보 열을 CLI로 지정받아야 합니다.

```
--quasi_identifiers COL [COL ...]     준식별자 열 (필수)
--sensitive_attributes COL [COL ...]  민감정보 열 (필수)
```

분모가 0인 경우(준식별자가 일치하는 원본 레코드가 없음)의 처리 방침을
반드시 명시하고 결과에 해당 레코드 수를 기록합니다.

**`inference_risk` (추론위험도, 식 (4))**

동일한 레코드가 없더라도 거리가 가까우면 확률적 추론이 가능하다는 점을 측정합니다.

```
p_hat = (1/n) * sum_i  I( d_i < d_or,i )

d_i    = 재현 레코드 i 와 가장 가까운 원본 레코드까지의 거리  (nearest_distance)
d_or,i = 그 원본 레코드와 가장 가까운 다른 원본 레코드까지의 거리 (reference_distance)
```

값이 1에 가까우면 추론 위험이 높고, 0에 가까우면 안전하지만 유용성이 떨어집니다.
**완벽한 재현 데이터의 이론값은 0.5** 이므로, 0.5에 가까울수록 유용성과 안전성이
균형 있게 생성된 것으로 봅니다. 결과 해석 시 이 기준을 함께 출력합니다.

`d_i == d_or,i` (동률)인 레코드는 분모 n 에서 제외하고 그 개수를 `n_ties` 로 보고합니다.
참고문헌 [1] 안내서 68쪽의 "인 경우의 수만큼 제외하고 나머지 n개의 비율을 계산함" 규칙입니다.

[2026-09-21 정정] `originality` 는 이 지표의 **옛 이름이며 같은 값**입니다.
`1 - inference_risk` 가 아니므로 `derived` 파생 필드를 두지 않습니다. (`CLAUDE.md` 용어 규칙 참조)

---

## 2. CLI 실행 구조

**4단계 실행 입도**를 둡니다.

```
sdqe <subcommand> [options]
```

| 서브커맨드 | 입도 | 설명 |
|---|---|---|
| `preprocess` | 단계 | 정합성 검증 + 초과 레코드 처리 |
| `threshold`  | 지표 단위 | `--metric` 으로 대상 지표 1개 선택 |
| `metric`     | 지표 단위 | `--name` 으로 개별 지표 1개 계산 |
| `utility`    | 그룹 | 유용성 지표 일괄 계산 |
| `privacy`    | 그룹 | 안전성 지표 일괄 계산 |
| `report`     | 전체 | 전처리부터 전 지표까지 실행 후 통합 리포트 생성 |

**파라미터 과다 문제 해결책**: 그룹/전체 모드는 `--config config.yaml` 로 설정을
주입받고, 개별 모드(`threshold`, `metric`)는 CLI 인자를 직접 받습니다.
둘이 함께 주어지면 CLI 인자가 우선합니다.

### 공통 인자

```
--original PATH          원본 데이터 경로 (필수)
--synthetic PATH         합성 데이터 경로 (필수)
--output_dir PATH        결과 저장 디렉터리 (기본: ./results)
--categorical_columns    범주형 변수 목록. 미지정 시 dtype으로 자동 추론
--exclude_columns        계산에서 제외할 변수 목록
--random_seed INT        기본 42
--n_jobs INT             병렬 처리 수. 기본 -1
--verbose                상세 로그 출력
```

### 사용 예시

```bash
# 전처리만
sdqe preprocess --original orig.csv --synthetic syn.csv \
                --reduction_method inference_distance --output_dir ./results

# 추론위험도 임계값 (시뮬레이션 방식)
sdqe threshold --metric inference_risk --method simulation \
               --n_simulations 100 --quantile 0.95 \
               --distance_metric gower --original orig.csv

# 구별위험도 임계값 (분포가정 방식, 계산 시간 단축)
sdqe threshold --metric single_out_risk --method distribution \
               --quantile 0.95 --original orig.csv

# 구별위험도 임계값을 0으로 엄격 지정
sdqe threshold --metric single_out_risk --strict --original orig.csv

# 안전성 지표 일괄
sdqe privacy --config config.yaml --linkage_threshold 0.7

# 전체 리포트
sdqe report --config config.yaml
```

임계값 계산은 **원본 데이터만으로 수행**됩니다(4.2절). `--synthetic` 은 필요 없습니다.

---

## 3. 전처리

### 3.1 정합성 검증

원본과 합성 데이터에 대해 다음을 확인하고, 불일치 시 명확한 에러 메시지를 냅니다.

- 변수 **이름** 일치 여부
- 변수 **순서** 일치 여부 (불일치 시 원본 순서로 자동 정렬, 정렬했음을 로그로 남김)
- 변수 **타입** 일치 여부 (범주형/수치형 구분 포함)
- 데이터 **형식** 일치 여부 (결측 표기, 범주 수준 집합 등)

### 3.2 초과 레코드 처리

합성 데이터가 원본보다 큰 경우를 처리하는 **독립 기능**으로 분리합니다.
현재 코드에도 구현되어 있으므로 참고합니다. 삭제 방식은 `--reduction_method`로 선택합니다.

| 값 | 방식 |
|---|---|
| `random` | 무작위 삭제 |
| `pmse_probability` | `pmse` 모델의 확률값을 계산해, 합성으로 판별될 확률이 높은(= 유용하지 않은) 레코드부터 삭제 |
| `inference_distance` | `inference_risk` 의 거리 비교 결과를 계산해, 미리 산출한 임계값 안으로 들어오도록 지시함수값이 0인 레코드와 1인 레코드를 적절한 비율로 삭제 |

전략 이름에 사용하는 신호(확률값/거리값)를 드러내어, 지표 이름과 혼동되지 않게 합니다.
기존 코드에서 이 전략이 `originality` 로 불렸다면 `inference_distance` 로 통일합니다.

### 3.3 캐시 연동 (중요)

`inference_distance` 방식은 전처리 **이전에** `inference_risk` 를 먼저 계산해야 합니다.
즉 이 방식으로 전처리하면 추론위험도는 이미 계산된 상태가 됩니다.

1. 전처리 과정에서 계산한 `nearest_distance`, `reference_distance`, `inference_risk`
   결과를 파일로 저장합니다.
2. 이후 `metric --name inference_risk` 실행 시,
   **동일한 입력 해시 + 동일한 거리 방식**의 결과가 이미 있는지 먼저 조회합니다.
3. 있으면 재사용하고 "캐시된 결과를 사용함"을 로그에 남깁니다. 없을 때만 새로 계산합니다.

입력 해시는 (원본 경로+수정시각, 합성 경로+수정시각, `distance_metric`, 관련 파라미터)로
만듭니다. 해시 생성 함수는 `io_utils.py` 한 곳에만 두고 양쪽이 같은 함수를 호출합니다.

거리 계산이 전체 계산 시간을 좌우하므로(4.6절), 이 캐시는 성능상 선택이 아니라 필수입니다.

---

## 4. 임계값 계산

근거: 참고문헌 [2] 3장. 핵심 아이디어는 **원본 데이터가 가장 완벽한 재현 데이터**라는
가정 아래, 원본 데이터만으로 귀무가설 하의 지표 분포를 만들어 분위수를 임계값으로 삼는 것입니다.

### 4.1 대상 지표

임계값 계산 대상은 **안전성 2종 + 유용성 2종**, 총 4종입니다.
`--metric` 인자로 하나를 선택합니다.

| 식별자 | 구분 | 근거 |
|---|---|---|
| `single_out_risk` | 안전성 | 논문 3.1, 3.2절 |
| `inference_risk` | 안전성 | 논문 3.1, 3.2절 |
| `bivariate_similarity` | 유용성 | 논문 범위 밖. 시뮬레이션 틀만 확장 적용 |
| `pmse` | 유용성 | 논문 범위 밖. 시뮬레이션 틀만 확장 적용 |

`linkage_risk`는 임계값 **계산 대상이 아닙니다.** 논문 3.2절은 CAP 임계값을
절대적 기준으로 볼 수 있으므로 별도 계산 없이 주관적으로 정하면 된다고 명시합니다.
(6.3절 참조) `univariate_similarity` 도 계산 대상이 아닙니다.

### 4.2 방식 1 — 시뮬레이션 (논문 3.1절)

원본 데이터를 이등분해 한쪽을 원본(A), 다른 쪽을 재현(B)으로 가정합니다.
재현으로 간주한 쪽도 사실은 원본이므로 유용성과 안전성이 보장된 데이터입니다.

```
단계 1: 원본 데이터를 50:50 크기로 무작위 분할하여 A(원본), B(재현)로 가정
단계 2: A와 B를 비교하여 대상 지표를 측정
단계 3: 위 과정을 충분히 반복한 뒤, 누적된 지표값의 분위수를 임계값으로 설정
```

- 반복 수는 임계값이 안정될 수 있는 수로, **경험적으로 100회 이상** 수행해야 합니다.
  `--n_simulations` 의 기본값을 100으로 두고, 100 미만이 입력되면 경고를 출력합니다.
- 분위수 의미를 `--help`와 실행 로그에 명시합니다.
  95% 선택은 실제로 안전한 재현 데이터를 5% 확률로 안전하지 않다고 판정한다는 뜻입니다.
  안전성 기준을 강화하려면 90%, 완화하려면 99%를 씁니다.
- 매 반복의 분할은 `--random_seed` 로부터 파생된 시드를 쓰고, 반복 인덱스별 시드를
  결과에 기록해 재현 가능하게 합니다.

#### 4.2.1 `single_out_risk` 크기 보정 (식 (5), 필수 구현)

시뮬레이션은 원본을 이등분하므로 양쪽 레코드 수가 각각 n/2입니다.
반면 실제 평가 상황에서는 원본과 같은 크기 n의 재현 데이터와 비교하는 경우가 많습니다.
레코드 수가 많아지면 동일 레코드가 존재할 확률이 증가하므로, 식 (2)의 값을
경험 분포와 직접 비교할 수 없고 다음 보정이 필요합니다.

```
p_star = 1 - (1 - p_hat)^2          ... 식 (5)

p_hat  : 원본을 절반으로 나누어 계산한 구별위험도
p_star : n개로 계산했을 경우의 보정값
```

- 이 보정은 **`single_out_risk` 에만** 적용합니다. `inference_risk` 에는 적용하지 않습니다.
- `--size_correction {on, off}` 로 전환 가능하게 하되 기본값은 `on` 입니다.
- 보정하지 않고 비교하려면 원본 데이터의 절반을 임의로 선택해 재현 데이터와
  비교해야 한다는 점을 `off` 선택 시 경고로 안내합니다.
- 결과 JSON에 `p_hat` 과 `p_star` 를 모두 기록합니다.

### 4.3 방식 2 — 분포 가정 (논문 3.2절)

시뮬레이션은 이등분 반복 계산 때문에 데이터가 커지면 계산량이 급증합니다.
분포 가정 방식은 **한 번의 계산**만으로 임계값을 얻어 시간을 크게 단축합니다.
적용 대상은 `single_out_risk` 와 `inference_risk` 두 가지입니다.

**`single_out_risk`**

```
단계 1: 원본 데이터를 이등분하지 않고, 첫 번째 레코드를 재현 레코드로 간주하여
        나머지 레코드에 동일 레코드가 존재하는지 계산한다.
        같은 방식으로 n개 레코드 전부에 대해 계산하여 식 (2)의 p_hat 을 구한다.
단계 2: p_hat ~ N( p, p(1-p)/n ) 을 이용해 p_hat 의 분포를 구한다.
단계 3: 위 정규분포에서 적절한 분위수를 선택해 임계값으로 지정한다.
```

**`inference_risk`**

```
단계 1: 첫 번째 원본 레코드 r_1 과 가장 가까운 레코드 r_k 사이 거리를 d_1,
        r_1 을 제외하고 r_k 와 가장 가까운 레코드와의 거리를 d_2 로 지정한다.
        d_1 < d_2 이면 1, 아니면 0으로 설정하는 방식으로 n개 레코드 전부에 대해 계산한다.
단계 2: p_A ~ N( p, p(1-p)/n ),  p = E(p_A) 를 따른다.
단계 3: 위 정규분포에서 적절한 분위수를 선택해 임계값으로 지정한다.
```

- 두 경우 모두 이항비율의 정규근사를 쓰므로, `n * p * (1-p)` 가 작으면
  근사가 부정확합니다. 값이 작을 때는 경고를 출력하고 시뮬레이션 방식을 권고합니다.
- `--quantile` 은 시뮬레이션과 동일한 인자를 재사용합니다(기본 0.95).
- 분포가정 방식에는 식 (5) 보정을 적용하지 않습니다. 이등분하지 않기 때문입니다.

### 4.4 방식별 지원 범위

지원하지 않는 조합은 명확한 에러로 거부합니다.
지원 여부는 하드코딩하지 말고 `config.py` 의 매핑 테이블 한 곳에서 관리합니다.

| 지표 | `simulation` | `distribution` | 기본값 |
|---|---|---|---|
| `single_out_risk` | 지원 | 지원 | `simulation` |
| `inference_risk` | 지원 | 지원 | `simulation` |
| `bivariate_similarity` | 지원 | 미지원 | `simulation` |
| `pmse` | 지원 | 미지원 | `simulation` |

```
Error: --method distribution is not supported for metric 'pmse'.
       Supported methods: simulation
```

### 4.5 지표별 추가 요구사항

**`single_out_risk`**
- 사용자가 임계값을 직접 `0`으로 엄격하게 설정할 수 있어야 합니다.
  `--strict` 플래그 또는 `--manual_threshold 0` 으로 제공합니다.
- `--strict` 가 주어지면 시뮬레이션이나 분포가정 계산을 수행하지 않고 즉시 0을 반환하며,
  `"threshold_source": "user_defined"` 로 기록합니다.

**`inference_risk`**
- 현재 코드에 구축된 **최적화 기능**을 유지하고 고려합니다. (4.6절)
- 거리 측정 방식을 `--distance_metric` 으로 선택합니다. (4.7절)

**`bivariate_similarity`, `pmse`**
- 시뮬레이션 방식만 지원하며, 식 (5) 보정은 적용하지 않습니다.
- 논문 범위 밖의 확장 적용이므로, 결과 JSON에
  `"note": "simulation framework extended beyond reference [2] scope"` 를 기록합니다.

### 4.6 계산 복잡도와 최적화

논문 4장에 따르면 시뮬레이션 방식에서 `inference_risk` 계산이 O(n^3)의 시간복잡도로
가장 많은 시간을 차지하며, 전체 계산 시간이 사실상 이 지표에 의해 결정됩니다.
분포 가정 방식은 복잡도는 같지만 반복이 없어 시간을 크게 단축합니다.

따라서 다음을 구현 방침으로 삼습니다.

- 최근접 거리 탐색은 이중 루프 대신 공간 분할 구조(KD-Tree, Ball-Tree) 또는
  청크 단위 행렬 연산으로 구현합니다.
- `reference_distance`(원본-원본 최근접 거리)는 원본 데이터에만 의존하므로
  **한 번 계산해 재사용**합니다. 시뮬레이션 반복마다 다시 계산하지 마십시오.
- 데이터 규모가 커서 분포 가정 방식이 유리한 상황이면 실행 시 그 사실을 안내합니다.
- **최적화로 수치가 달라져서는 안 됩니다.** 소규모 데이터에서 단순 구현과
  최적화 구현의 결과가 일치하는지 검증하는 테스트를 반드시 작성합니다.

### 4.7 거리 측정 방식 (임계값/지표 공통)

`common/distance.py`에 구현하여 임계값 계산과 `inference_risk` 계산이 **같은 함수**를 씁니다.

| 값 | 방식 |
|---|---|
| `gower` | Gower distance. 범주형은 일치 여부, 수치형은 min-max 스케일링 후 차이로 계산 |
| `cosine` | 범주형 변수를 임의 label encoding한 뒤 코사인 거리로 계산. GPU 고속 연산 가능 |

`cosine` 방식은 `--use_gpu` 플래그로 GPU 사용을 제어하고, 라이브러리가 없으면
경고 후 CPU로 자동 폴백합니다.

거리 방식이 바뀌면 `inference_risk` 값과 임계값이 모두 달라집니다.
**지표 계산과 임계값 계산에 서로 다른 거리 방식이 쓰이면 비교가 무의미하므로,
판정 단계에서 두 값의 `distance_metric` 이 같은지 확인하고 다르면 에러로 중단합니다.**

### 4.8 결과 관리

- 계산된 임계값은 터미널에 평문으로 출력하고 JSON으로도 저장합니다.
- **시뮬레이션 결과는 전체 값을 보존합니다.** 100회 반복이면 100개 값을 모두 남깁니다.
  요약 통계(평균, 표준편차, 분위수)는 별도 필드로 함께 기록합니다.
- 원시 배열은 `<metric>_simulations.json` 으로 분리 저장하고,
  요약 JSON에서는 파일 경로만 참조합니다.
- 분포 가정 방식은 추정된 정규분포의 평균과 분산, 사용한 n을 기록합니다.

---

## 5. 유용성 지표

최우선 고려사항은 **최적화와 정확한 계산 방식**입니다.

### 5.1 `univariate_similarity` (1차원 유사성)

변수별 분포를 비교하는 일반적인 함수입니다. 임계값 계산 대상이 아닙니다.

- 각 변수마다 plot을 그립니다. (현행 유지)
- **각 plot 아래에 분포 비교 검정 결과값을 텍스트로 기입합니다.**
  - 수치형 변수: KS test 통계량과 p-value
  - 범주형 변수: chi-square test 통계량과 p-value
- 추가로, **전체 변수를 한 장으로 비교할 수 있는 총정리 plot 1장**을
  적절한 레이아웃(변수 개수에 따라 행/열 자동 계산)으로 그립니다.
- 한글 변수명이 입력될 수 있으므로 **한글 폰트 설정 코드**가 반드시 필요합니다.
  폰트를 찾지 못하면 경고를 남기고 깨지지 않는 대체 폰트로 진행합니다.

### 5.2 `bivariate_similarity` (2차원 유사성, OLS)

**해결해야 할 문제**
- OLS 표현식(formula) 방식은 변수명에 띄어쓰기나 특수문자가 들어가면 에러가 납니다.
- 상수항 변수(예: 모든 값이 `1`인 변수)가 있으면 에러가 납니다.

**요구 구현**
- 문자열 formula에 의존하지 않도록, 설계행렬(design matrix)을 직접 구성합니다.
  범주형은 직접 더미 인코딩하고, 변수명은 내부 안전 식별자로 매핑한 뒤
  결과 출력 시 원래 이름으로 복원합니다.
- 다른 패키지(`numpy.linalg.lstsq`, `sklearn.linear_model` 등)를 써도 무방하되,
  계산 결과가 기존 OLS와 동일함을 확인합니다.
- 상수항 변수는 `--drop_constant_columns` 옵션으로 제외할 수 있게 하고,
  제외된 변수 목록을 결과 JSON에 기록합니다.
- 이 지표 자체를 제외하고 실행할 수 있는 선택지도 제공합니다.

### 5.3 `pmse`

모델을 직접 생성하여 확률값으로 계산합니다.

- 기존 코드는 로지스틱 회귀만 구현되어 있습니다. 대표 모델 **3가지**를 선택할 수 있게 합니다.
  - `logistic_regression`(기본), `decision_tree`, `gradient_boosting`
  - [2026-09-18 변경] 기존 Kdata 코드는 실제로 로지스틱 회귀(sklearn 기본 설정)만 사용했음.
    깊이 제한 없는 결정트리는 과적합으로 pMSE 가 0.25 로 퇴화하므로 기본값을 로지스틱 회귀로 변경.
  - `--model` 인자로 선택
- 세부 파라미터는 **필수 입력이 아니지만 설정 가능**해야 합니다.
  - 예: `--model_params '{"max_depth": 5, "min_samples_leaf": 10}'` (JSON 문자열)
- 3.2절 `pmse_probability` 삭제 전략이 여기서 산출한 확률값을 재사용하므로,
  확률값 산출 함수를 별도로 분리해 두어야 합니다.
- 모델 학습 시 `random_state`에 `--random_seed` 값을 반드시 전달합니다.

---

## 6. 안전성 지표

지표 정의와 수식은 1.2절을 따릅니다.

### 6.1 `single_out_risk` (구별위험도)

- 식 (2)를 기본으로 구현하며, `--duplicate_adjustment off` 로 식 (1)도 선택 가능합니다.
- 계산 전에 변수 순서와 명칭이 통일되어 있는지만 확인합니다.
  확인 로직은 3.1절 정합성 검증 유틸을 재사용합니다.
- 대규모 데이터에서는 행 단위 비교 대신 해시 기반 조인으로 처리해 속도를 확보합니다.
  원본 내 중복 수 `d_i` 도 같은 그룹화 연산에서 함께 얻습니다.
- 임계값과 비교할 때, 임계값이 시뮬레이션 방식으로 산출되었고 식 (5) 보정이
  적용되었는지 확인합니다. 보정 여부가 다르면 경고합니다.

### 6.2 `inference_risk` (추론위험도)

- 식 (4)로 계산하며, 이론 기준값 0.5를 결과에 함께 출력합니다.
  값이 0.5보다 크면 추론 위험이 높은 쪽, 작으면 유용성이 낮은 쪽임을 안내합니다.
- 거리 측정 방식을 `--distance_metric`(`gower` | `cosine`)으로 선택합니다.
  구현은 `common/distance.py` 의 공용 함수를 호출합니다. (4.7절)
- **계산 전에 3.3절의 캐시 조회 절차를 반드시 거칩니다.**
- 동률(`d_i == d_or,i`)인 레코드는 분모에서 제외하고 `n_ties`, `n_evaluated` 를 함께 기록합니다.
- `originality` 는 이 지표의 옛 이름이므로 별도 필드로 기록하지 않습니다. [2026-09-21 정정]

### 6.3 `linkage_risk` (연결위험도, CAP)

- 기존 `cap_evaluation_.py` 의 로직을 `metrics/privacy/` 로 이전합니다.
  **이전 후 계산 결과는 기존과 동일해야 합니다.** 구조만 정리하고 수식은 바꾸지 마십시오.
- `--quasi_identifiers` 와 `--sensitive_attributes` 를 필수 인자로 받습니다.
- 임계값은 계산하지 않고 **사용자가 직접 지정**합니다.

```
--linkage_threshold FLOAT    연결위험도 판정 임계값 (기본 0.7, 허용 범위 0.0 이상 1.0 미만)
```

- 논문 3.2절 근거: CAP 임계값이 1이면 준식별자로 민감정보를 100% 맞출 수 있으므로
  **1 미만의 값**을 선택해야 합니다. 1에 가까우면 연결 위험이 커지고, 너무 작게 잡으면
  기준을 넘는 레코드가 많아져 이를 삭제할 때 유용성이 떨어집니다.
  따라서 1.0 이상의 값은 에러로 거부하고, 0.5 미만이면 유용성 저하 경고를 출력합니다.
- 결과 JSON에는 `"threshold_source": "user_defined"` 를 기록하고,
  임계값을 넘는 레코드 수와 비율을 함께 보고합니다.

---

## 7. 결과 파일 규격

모든 결과는 `--output_dir` 아래에 저장합니다. 파일명은 지표 식별자를 그대로 씁니다.

```
results/
├── preprocess/
│   ├── validation.json
│   └── reduction.json
├── threshold/
│   ├── single_out_risk.json
│   ├── single_out_risk_simulations.json
│   ├── inference_risk.json
│   ├── inference_risk_simulations.json
│   ├── bivariate_similarity.json
│   └── pmse.json
├── metrics/
│   ├── univariate_similarity.json
│   ├── bivariate_similarity.json
│   ├── pmse.json
│   ├── single_out_risk.json
│   ├── inference_risk.json
│   └── linkage_risk.json
├── figures/
│   ├── univariate_similarity_<변수명>.png
│   └── univariate_similarity_summary.png
├── cache/
│   └── <hash>.json
└── report.json
```

각 JSON은 공통으로 다음 필드를 포함합니다.

```json
{
  "metric": "inference_risk",
  "value": 0.4812,
  "reference_value": 0.5,
  "threshold": 0.45269,
  "threshold_source": "simulation",
  "passed": true,
  "parameters": {
    "distance_metric": "gower",
    "n_simulations": 100,
    "quantile": 0.95,
    "random_seed": 42
  },
  "input_info": { "n_original": 48842, "n_synthetic": 48842, "n_columns": 13 },
  "excluded_columns": [],
  "cache_hit": false,
  "computed_at": "2026-09-18T10:00:00+09:00",
  "elapsed_seconds": 12.4
}
```

- `reference_value` 는 이론 기준값이 있는 지표(`inference_risk`)에만 씁니다.
- `single_out_risk` 의 시뮬레이션 임계값 JSON에는 `p_hat` 과 `p_star`(식 5 보정값),
  `size_correction` 적용 여부를 함께 기록합니다.

---

## 8. 검증용 참조값

참고문헌 [2] 4장의 실험 결과입니다. 구현 검증용 회귀 테스트 기준으로 사용합니다.

**실험 조건**
- 데이터: UCI Machine Learning Repository, Adult 데이터셋 (참고문헌 [9])
- 규모: 48,842개 관측치, 13개 변수
- 준식별자: 연령, 고용 형태, 결혼 상태, 인종, 성별 / 민감정보: 소득 분류
- 시뮬레이션: 50:50 무작위 분할, 100회 반복, 95분위수

| 지표 | 시뮬레이션 방식 | 분포 가정 방식 |
|---|---|---|
| `single_out_risk` | 0.19820 | 0.20562 |
| `inference_risk` | 0.45269 | 0.51066 |

- 동일 조건에서 구현 결과가 이 값에 근접하는지 확인합니다.
  무작위 분할 때문에 시뮬레이션 값은 정확히 일치하지 않으므로, 허용 오차를 두고
  비교하거나 분포 가정 방식의 값으로 검증하십시오.
- 두 방식의 `inference_risk` 값 차이(0.45 대 0.51)는 계산 절차가 다르기 때문입니다.
  이 차이를 버그로 오인하지 마십시오.
- 이 데이터셋은 공개 데이터이므로 테스트 픽스처로 사용할 수 있습니다.

---

## 9. 미확정 사항 (확인 필요)

구현 전에 작성자 확인이 필요한 항목입니다. **임의로 결정하지 마십시오.**

1. **기존 코드의 `SingleOut` / `Originality` 실제 구현 위치**
   논문 기준으로 `SingleOut` 은 구별위험도(식 2), `Originality` 는 추론위험도(식 4)
   계열입니다. 그런데 기존 요구사항 문서에는 두 용어가 모두 추론위험도의 하위
   구성요소로 적혀 있었습니다.
   현재 코드에서 각 함수가 실제로 어떤 수식을 계산하는지 확인한 뒤,
   1.1절 식별자 체계로 이름을 통일해야 합니다.
   **코드를 읽고 수식을 대조한 결과를 먼저 보고하십시오.**

2. **참고문헌 [1] 서지사항**
   10장 [1]이 비어 있습니다. 논문 참고문헌 [3]의
   개인정보보호위원회 '합성데이터 생성 참조모델'(2024.5)이 이에 해당하는지
   확인이 필요합니다.

3. **`linkage_risk` 분모 0 처리**
   준식별자가 일치하는 원본 레코드가 없을 때의 처리 방침(제외, 0으로 처리 등)이
   기존 `cap_evaluation_.py` 에서 어떻게 되어 있는지 확인하고 그대로 유지합니다.

### 해결된 사항 (참고)

- **[9장 1번, 2026-09-18 해결]** 기존 코드 대조 결과: `SingleOut`(`compute_singleout`)은 식 (1) 단순 일치 비율,
  현행 `Originality`(`compute_originality_matrix`)는 식 (4)가 아닌 "원본+합성 전체에서 최근접 이웃이 원본인 비율"
  (풀링 방식)이었음. 구버전 `_Metric_/RiskMetric.py` 의 `_get_dsdo_safe` 가 식 (4)에 해당.
  작성자 결정: `inference_risk` 는 식 (4)로 구현, `cosine` 거리는 기존 혼합 방식 유지,
  `bivariate_similarity` 의 (범주형, 수치형) 쌍 Cramér's V 버그는 수정.
- **[9장 2번 해결]** docs 의 참고문헌1 은 '합성데이터 생성 참조모델'(2024.5)이 아니라
  개인정보보호위원회 「합성데이터 생성·활용 안내서」(2024.12)임. 10장 [1] 수정.
- **[9장 3번 해결]** `cap_evaluation_.py` 는 **원본 레코드마다** 합성에서 일치 수를 세며,
  분모 0(준식별자 일치 합성 레코드 없음)이면 CAP=0 으로 평균에 포함하고 `unmatched_count` 로 보고.
  작성자 결정: 기존 방식 그대로 유지 (명세 1.2절의 "재현 레코드마다" 와 방향이 다름에 유의).
- 임계값 대상 지표: 안전성 2종 + 유용성 2종 (4.1절)
- 분포가정 적용 범위: `single_out_risk`, `inference_risk` 만 (4.3, 4.4절)
- 분포가정의 구체적 분포: 이항비율의 정규근사 `N(p, p(1-p)/n)` (4.3절)
- 시뮬레이션 크기 보정식 (5): `single_out_risk` 에만 적용 (4.2.1절)
- 연결위험도 임계값: 사용자 지정, 기본 0.7, 1.0 미만 (6.3절)
- 패키지/명령어 이름: `sdqe`
- 용어 혼용: 1.1절 표로 고정 (`CLAUDE.md` 용어 규칙)

---

## 10. 참고문헌

- [1] 개인정보보호위원회, 「합성데이터 생성·활용 안내서」, 2024. 12.
- [2] 박민수, 정미령, 김승환, 박헌진, "재현 데이터 안전성 평가 지표의 임계값 연구",
  *Journal of Korean Institute of Intelligent Systems*, vol. 34, no. 6, 2024.
  http://dx.doi.org/10.5391/JKIIS.2024.34.6.000
  → 안전성 지표 정의(식 1~4)와 임계값 계산 방식(시뮬레이션, 분포가정, 식 5)의 근거
- [3] Article 29 Data Protection Working Party, "Opinion 05/2014 on anonymisation
  techniques", 2014. → Singling out / Linkability / Inference 세 가지 위험 기준
- [4] Taub, J. and Elliot, M., "The synthetic data challenge",
  *UNECE: Conference of European Statisticians*, 2019. → CAP
- [5] Stan, M., Jordi, N., Morvarid, S., and Tomasz, S., "A review of attribute
  disclosure control", *Advanced Research in Data Privacy*, vol. 567, pp. 41-61, 2015.
- [6] Gower, J. C., "A General Coefficient of Similarity and Some of Its Properties",
  *Biometrics*, vol. 27, no. 4, pp. 857-871, 1971. → Gower distance
- [7] Woo, M.-J., Reiter, J. P., Oganian, A., and Karr, A. F., "Global Measures of Data
  Utility for Microdata Masked for Disclosure Limitation",
  *Journal of Privacy and Confidentiality*, vol. 1, no. 1, 2009. → pMSE
- [8] Becker, B. and Kohavi, R., "Adult, UCI Machine Learning Repository", 1996.
  https://doi.org/10.24432/C5XW20 → 8장 검증용 데이터셋
