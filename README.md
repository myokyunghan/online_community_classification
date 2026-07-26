# 온라인 커뮤니티 연구지형 분류

"온라인 커뮤니티" 키워드로 수집된 학술 논문들을 미리 정의한 분류 체계(taxonomy)에 따라
자동 분류하는 파이프라인이다. NVIDIA NIM(build.nvidia.com)의 무료 API를 통해 LLM에게
논문 제목/저자/출처/초록을 주고, 정해진 판별 기준에 따라 최대 2개까지 코드를 매기게 한다.

## 분류 체계

논문 한 편에는 **역할적 특성(ROLE)** 과 **환경적 특성(ENV)** 두 축의 코드가 각각 독립적으로
매겨지며, 두 축을 합쳐서 확률 상위 **최대 2개**까지만 최종 채택된다. 정확한 판별 기준 문장은
`prompt_template.txt`를 참고할 것 — 아래는 요약이다.

**ROLE (역할적 특성)** — 온라인 커뮤니티를 논문이 "무엇으로" 쓰는가

| 코드 | 의미 |
|---|---|
| `ROLE_FIELD_PUBLICSPHERE` | 온라인 커뮤니티 자체가 공론장으로 기능 — 이용자 간 상호작용(댓글, 의견 교환)이 분석 대상 |
| `ROLE_FIELD_SOCIALCAPITAL` | 온라인 커뮤니티를 신뢰·연대·네트워크 같은 관계적 자원이 형성되는 공간으로 분석 (관계 상대방도 같은 커뮤니티 이용자여야 함) |
| `ROLE_WINDOW_POLARIZE` | 정치적 양극화·진영 갈등을 보여주는 사례로 온라인 커뮤니티를 사용 |
| `ROLE_WINDOW_GENDER` | 젠더 갈등·페미니즘 대립 담론을 보여주는 사례로 사용 |
| `ROLE_WINDOW_HATE` | 혐오 표현·정서 확산을 보여주는 사례로 사용 |
| `ROLE_WINDOW_MISC` | 위 세 가지 외의 다른 사회현상(결혼문화, 팬덤, 노동서사 등)을 보여주는 사례로 사용 |

**ENV (환경적 특성)** — 온라인 커뮤니티의 어떤 제도/구조적 조건을 비교·분석하는가

| 코드 | 의미 |
|---|---|
| `ENV_COMMUNITY_SUBSCRIPTION` | 회원제 유무·등급 차이에 따른 결과 차이 |
| `ENV_COMMUNITY_GOVERNANCE` | 운영·제재·신고 규칙이 다른 사례/시기 비교 |
| `ENV_COMMUNITY_DEMOGRAPHIC` | 커뮤니티 실제 구성원의 인구통계 구성(연령·성별·지역·소득)이 분석 초점 |
| `ENV_COMMUNITY_LIFECYCLE` | 커뮤니티 자체의 생애주기(형성/분화/쇠퇴)와 그 결과가 분석 초점 |
| `ENV_ONLINE_ANNONIMITY` | 익명성 수준이 다른 조건의 비교 |

어느 코드에도 해당하지 않으면(예: 온라인 커뮤니티가 다른 연구의 운영 수단으로만 언급된 경우)
`ERR` 상태로 분류된다.

## 사전 준비

```bash
pip install -r requirements.txt        # openai, pandas, openpyxl
export NVIDIA_API_KEY="nvapi-..."      # build.nvidia.com에서 무료 발급
```

### NVIDIA API 키 발급 방법

1. https://build.nvidia.com 접속 후 계정으로 로그인(없으면 가입, 무료).
2. 우측 상단 프로필 아이콘 → **API Keys** 메뉴로 이동.
3. **Generate API Key**(또는 **Create New Key**) 클릭 → 키가 `nvapi-`로 시작하는 문자열로 발급됨.
4. 발급된 키는 그 화면에서만 전체가 보이므로 바로 복사해서 안전한 곳에 저장해둔다.
5. 터미널에서 아래처럼 환경변수로 등록한다 (매번 새 터미널을 열 때마다 다시 설정해야 하므로,
   계속 쓸 거면 `~/.zshrc`나 `~/.bash_profile`에 추가해두면 편하다):

   ```bash
   export NVIDIA_API_KEY="nvapi-여기에_발급받은_키"
   ```

무료 등급에는 호출 횟수 제한이 있을 수 있으니, 693편처럼 많은 논문을 한 번에 돌릴 때
`--limit`으로 소량 테스트 후 전체를 실행하는 것을 권장한다.

## 파일 구조

| 파일 | 역할 |
|---|---|
| `prompt_template.txt` | 분류 판별 기준 전문 (LLM에게 주는 지침). 분류 규칙을 바꾸려면 이 파일만 수정하면 된다. |
| `classify_paper.py` | 핵심 모듈. NVIDIA API 호출, 응답 JSON 파싱(`extract_json`), role/env 코드를 audit 배열로부터 기계적으로 산출(`derive_role_env_from_audit`)하는 로직이 들어있다. 다른 스크립트가 전부 이 모듈을 가져다 쓴다. |
| `compare_with_golden.py` | **정답이 있는 22편**(`golden/` 폴더)을 대상으로 모델 예측과 정답을 비교해 정확도를 측정하는 평가용 스크립트. 프롬프트를 수정한 뒤 회귀 확인용으로 쓴다. |
| `classify_targets.py` | **정답이 없는 실제 논문**(`target/` 폴더의 엑셀)을 분류해 결과 CSV를 만드는 실사용 스크립트. 아래 "실제 논문 분류" 참고. |
| `inspect_paper.py` | golden 22편 중 특정 id 하나만 뽑아 모델의 전체 판단 근거(audit 전체, quote, prob)를 JSON으로 출력. 특정 논문이 왜 오분류됐는지 디버깅할 때 사용. |
| `find_mismatches.py` | `compare_with_golden.py` 실행 후 나온 `comparison_results.csv`에서 오분류(불일치)만 골라 `golden/mismatches.csv`로 뽑아준다. |
| `golden/온라인_커뮤니티_연구지형_분류논문_22편.csv` | 정답이 붙어있는 22편 + 판단불가 테스트케이스 1편(id=23, class1=ERR). 프롬프트 튜닝/회귀 검증용 기준 데이터셋. |
| `target/KCI 사회과학.xlsx` | 실제로 분류해야 하는 논문 원본 데이터 (KCI 다운로드, 정답 없음). |

## golden 데이터셋이란?

`golden/온라인_커뮤니티_연구지형_분류논문_22편.csv`는 실제 문헌고찰 논문에서 이미 사람이
분류해놓은 22편(+판단불가 테스트케이스 1편)을 정답으로 담아둔 기준 데이터셋이다. 목적은
"이 논문들을 분류하는 것" 자체가 아니라, **모델이 정답을 얼마나 잘 맞히는지 측정해서
`prompt_template.txt`의 판별 기준이 제대로 작동하는지 검증**하는 것이다. 즉, `target/` 폴더의
실제 논문들과는 성격이 다르다 — golden은 정답이 있는 "채점용" 데이터, target은 정답이 없는
"실사용" 데이터다.

컬럼 구성:
- `id`, `author`, `published_year`, `title`, `journal`, `abstract`: 논문 기본 정보
- `class1`, `class2`: 사람이 매긴 정답 코드. 우선순위 없는 집합이라 순서는 의미가 없다
  (class1에 SOCIALCAPITAL, class2에 PUBLICSPHERE여도 그 반대여도 같은 정답으로 취급).
  두 코드 다 붙는 경우도, class2가 비어 코드 1개짜리인 경우도 있다.
- id=23은 실제로는 "온라인 커뮤니티 연구가 아닌" 논문을 넣어둔 특수 케이스로, `class1`이
  문자 그대로 `"ERR"`이다 — 모델이 11개 코드 중 아무것도 부여하지 않고 정확히 `ERR`로
  응답하는지를 테스트한다.

이 golden 셋을 쓰는 흐름은 이렇다: `prompt_template.txt`의 판별 기준을 수정 →
`compare_with_golden.py` 실행해서 22편에 대한 정확도 확인 → 틀린 논문이 있으면
`find_mismatches.py`로 뽑아서 원인 분석 → `inspect_paper.py --id N`으로 해당 논문 하나의
모델 판단 근거(quote/prob)를 자세히 들여다보고 → 필요하면 다시 프롬프트를 수정. 이 반복으로
현재 `prompt_template.txt`의 판별 기준이 다듬어졌다.

`compare_with_golden.py`가 만드는 `golden/comparison_results.csv`의 주요 컬럼:
- `exact_match`: 예측 코드 집합이 정답 집합과 완전히 같은지
- `gold_covered`: 예측이 정답을 전부 포함하는지 (추가 예측은 허용하는 느슨한 기준)
- `gold_dropped_by_cap`: "합쳐서 최대 2개" 캡 때문에 정답 코드가 잘려나갔는지
- `majority_status`, `majority_codes`: `--runs-per-paper`를 여러 번 줬을 때 다수결로 채택된 최종 상태/코드
- `full_agreement`: 여러 번 호출한 결과가 전부 똑같았는지 (모델 응답의 안정성 지표)

## 실행 방법

### 1. 실제 논문 분류 (production 용도)

`target/KCI 사회과학.xlsx`에 있는 논문들을 분류해서 `target/classification_results.csv`를 만든다.

```bash
python3 classify_targets.py                    # 전체 실행
python3 classify_targets.py --limit 10          # 먼저 10편만 테스트
python3 classify_targets.py --runs-per-paper 3  # 논문당 3회 호출 후 다수결로 안정화
```

특징:
- 초록(한글/영문 둘 다)이 없는 논문은 API를 호출하지 않고 `NO_ABSTRACT`로 바로 기록한다.
- 논문 한 편 처리할 때마다 즉시 결과 파일에 저장하므로, 중간에 끊겨도 데이터가 남는다.
- 다시 실행하면 이미 `classification_results.csv`에 있는 논문은 자동으로 건너뛰고 이어서 처리한다.
  전체를 처음부터 다시 하려면 `--overwrite`를 붙인다.

결과 CSV의 주요 컬럼:
- `status`: `OK`(분류됨) / `ERR`(온라인 커뮤니티 연구 아님) / `NO_ABSTRACT`(초록 없어 스킵) / `FAILED`(API 오류로 처리 실패)
- `role_codes`, `env_codes`: 최종 채택된 코드(합쳐서 최대 2개)
- `focus`, `rationale`: 모델이 왜 그렇게 판단했는지의 요약
- `dropped_by_cap`: 최대 2개 캡 때문에 잘려나간 코드가 있었는지
- `run_details`, `full_agreement`, `error_runs`: `--runs-per-paper`를 2 이상으로 줬을 때 각 회차 결과와 회차 간 합의 여부

### 2. golden 데이터셋 기반 테스트 (프롬프트 수정 시 회귀 확인용)

가장 기본적인 실행:

```bash
python3 compare_with_golden.py
```

`golden/comparison_results.csv`에 논문별 예측/정답/일치 여부가 저장되고, 콘솔에 전체 정확도
요약(완전 일치, 정답 포함 여부, 캡으로 잘린 정답, ERR 케이스 정답 여부 등)이 출력된다.

자주 쓰는 옵션 조합:

```bash
python3 compare_with_golden.py --limit 3                 # 앞 3편만 빠르게 확인 (프롬프트 수정 직후 감 잡을 때)
python3 compare_with_golden.py --sample 5 --seed 1        # 22편 중 무작위 5편 (재현 가능하게 시드 고정)
python3 compare_with_golden.py --runs-per-paper 3          # 논문당 3회 호출 후 다수결로 안정성까지 확인
python3 compare_with_golden.py --output golden/tmp.csv     # 결과를 다른 파일로 저장 (기존 결과와 비교하고 싶을 때)
```

`compare_with_golden.py`가 지원하는 전체 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--csv` | `golden/온라인_커뮤니티_연구지형_분류논문_22편.csv` | 정답 CSV 경로 |
| `--output` | `golden/comparison_results.csv` | 결과 저장 경로 |
| `--api-key` | `NVIDIA_API_KEY` 환경변수 | API 키 직접 지정 |
| `--limit N` | 없음 | 앞 N편만 처리 |
| `--sample N` | 없음 | 무작위 N편만 처리 |
| `--seed N` | 없음 | `--sample`의 무작위 시드 (재현성용) |
| `--runs-per-paper N` | 1 | 논문당 반복 호출 횟수, 과반수로 최종 채택 |
| `--sleep N` | 1.0 | 호출 사이 대기 시간(초) |
| `--max-retries N` | 3 | 오류/파싱 실패 시 최대 재시도 횟수 |
| `--retry-wait-minutes N` | 3.0 | 서버 오류 재시도 전 대기 시간(분) |

프롬프트를 수정했을 때의 일반적인 확인 흐름:

```bash
# 1) 수정한 판별 기준이 방향은 맞는지 3~5편으로 빠르게 확인
python3 compare_with_golden.py --sample 5 --seed 1

# 2) 문제 없어 보이면 22편 전체로 정확도 확인
python3 compare_with_golden.py

# 3) 안정성까지 보고 싶으면 다수결로 재확인
python3 compare_with_golden.py --runs-per-paper 3
```

### 3. 평가 결과에서 오분류만 뽑기

```bash
python3 find_mismatches.py
```

`compare_with_golden.py`를 먼저 실행해둔 상태에서, 정답과 어긋난 논문(`exact_match=False`)만
초록·판단근거와 함께 `golden/mismatches.csv`로 뽑아낸다. 어떤 논문들이 왜 틀렸는지 한눈에
훑어볼 때 쓴다.

### 4. 특정 논문 하나 깊이 디버깅

```bash
python3 inspect_paper.py --id 4
python3 inspect_paper.py --id 4 --output golden/inspect_4_retest.json   # 다른 파일명으로 저장
```

golden CSV의 id=4 논문에 대해 모델이 11개 코드 전부에 매긴 확률(prob)과 근거 문장(quote)을
`golden/inspect_<id>.json`에 저장한다. `find_mismatches.py`로 어떤 논문이 틀렸는지 찾은 뒤,
왜 그 코드가 부여/미부여됐는지 근거 문장 단위로 파고들 때 쓴다.

## 참고: 모델과 한계

- 사용 모델은 `classify_paper.py`의 `MODEL` 상수(`qwen/qwen3-next-80b-a3b-instruct`)에
  고정되어 있다. 다른 모델(`meta/llama-3.3-70b-instruct`, `meta/llama-4-maverick-17b-128e-instruct`
  등)로도 테스트해봤으며, 모델을 바꿀 때는 이 상수만 수정한 뒤 반드시
  `compare_with_golden.py`로 golden 22편 정확도가 떨어지지 않는지 확인해야 한다 — 모델마다
  이 복잡한 JSON 스키마(11개 코드 전수 평가 + prob + quote)를 따르는 안정성이 다르다.
- `temperature=0`으로 호출하지만 완전히 결정적이지는 않다. 안정성이 중요하면
  `--runs-per-paper`를 3 이상으로 주고 다수결(`majority_codes`)을 신뢰하는 것을 권장한다.
- `prompt_template.txt`의 판별 기준은 golden 22편에 대해 반복적으로 튜닝된 결과이며, 완벽하지
  않다 — 특정 코드(예: `ROLE_FIELD_PUBLICSPHERE`)가 간헐적으로 과다 부여되는 경향이 남아있다.
  분류 규칙을 조정하고 싶다면 `prompt_template.txt`만 수정한 뒤 `compare_with_golden.py`로
  회귀를 확인하면 된다.
