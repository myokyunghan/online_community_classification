# ICR 분류 파이프라인

온라인 커뮤니티 관련 논문을 11개(실제로는 `ENV_COMMUNITY_DEMOGRAPHIC`을 제외한 10개) 코드로
분류하는 파이프라인. NVIDIA NIM API(`openai/gpt-oss-120b`)를 통해 호출한다.

## 폴더 구조

```
ICR/
├── README.md                                    이 파일
├── 온라인_커뮤니티_연구지형_분류논문_30편.csv        golden 데이터셋(정답 30편)
│
├── prompt_template_division.txt                 1차 호출: 커뮤니티 존재 + 역할/환경 관련 여부 판단
├── prompt_template_role.txt                      2차 호출: ROLE 6개 코드 채점 (role_relevant일 때만)
├── prompt_template_env.txt                       3차 호출: ENV 4개 코드 채점 (env_relevant일 때만)
│
├── classify_paper_fewshot.py                     분류 로직 본체 (1~3차 호출 오케스트레이션)
├── compare_with_golden_fewshot.py                golden CSV 기준 배치 평가 스크립트
├── repeat_eval.py                                 배치 평가를 N번 반복해 평균/표준편차 계산
│
├── prompt_template_system.txt                    (더 이상 안 씀 — 아래 "사용하지 않는 파일" 참고)
└── comparison_results_fewshot*.csv               평가 실행 결과물 (스크립트가 생성)
```

## 분류 코드 (10개)

| 축 | 코드 |
|---|---|
| ROLE - FIELD(장) | `ROLE_FIELD_PUBLICSPHERE`, `ROLE_FIELD_SOCIALCAPITAL` |
| ROLE - WINDOW(창) | `ROLE_WINDOW_POLARIZE`, `ROLE_WINDOW_GENDER`, `ROLE_WINDOW_HATE`, `ROLE_WINDOW_MISC` |
| ENV | `ENV_COMMUNITY_SUBSCRIPTION`, `ENV_COMMUNITY_GOVERNANCE`, `ENV_COMMUNITY_LIFECYCLE`, `ENV_ONLINE_ANNONIMITY` |

어떤 코드에도 해당하지 않으면 `ERR`.

## 파이프라인 구조 (왜 3번에 나눠 호출하는가)

논문 1편을 판정할 때 최대 3번의 API 호출이 순차적으로 일어난다. 10개 코드를 한 번에 다 판단하게
하면 부담이 크고, 그러다 보면 판단력이 떨어지는 문제가 있어서 이렇게 나눴다.

```
1차: division  (has_community, role_relevant, env_relevant 판단)
       │
       ├─ has_community=false 이거나 role/env 둘 다 false ──▶ 즉시 ERR, 종료 (호출 끝)
       │
       ├─ role_relevant=true ──▶ 2차: role  (창/장 반사실 테스트 + ROLE 6개 채점)
       │
       └─ env_relevant=true  ──▶ 3차: env   (ENV 4개 채점)

2차·3차에서 나온 audit을 합쳐서(최대 10개) 최종적으로 prob≥0.5인 것 중 상위 2개만 채택한다.
("합쳐서 최대 2개" 캡은 ROLE/ENV를 나눠 호출해도 전체 10개 기준으로 한 번에 적용된다.)
```

입력은 **초록만** 준다(제목·저자·출처는 제외). 제목에 "온라인 커뮤니티"가 언급되는 것만 보고
실제 초록엔 관련 내용이 없는데도 `has_community`를 잘못 true로 판단하는 현상이 있어서, 초록
내용만으로 판단하게 했다. (단, 이 때문에 반대로 제목에만 있던 맥락이 사라져 손해를 보는
사례도 있었다 — "알려진 한계" 참고.)

few-shot 예시는 **대화 턴(user/assistant)으로 주고받지 않고, 시스템 프롬프트 안에 텍스트로
그대로 적어 넣는다.** 실제 호출은 항상 `system` 1개 + `user`(논문) 1개, 단일 왕복이다 —
멀티턴으로 주고받으면 이 모델(`openai/gpt-oss-120b`)이 `finish_reason='stop'`인데도
`content`가 비는 현상이 반복 확인됐다.

## 사전 준비

```bash
export NVIDIA_API_KEY="nvapi-..."
```

## 사용법

### 1. 논문 한 편만 분류

```bash
python3 ICR/classify_paper_fewshot.py --id 1
python3 ICR/classify_paper_fewshot.py --paper-file paper.txt
```

### 2. golden 30편 기준 배치 평가

```bash
# 기본: 무작위 3편을 few-shot으로 뽑고(ERR 사례 1개는 항상 포함), 나머지 전체 평가
python3 ICR/compare_with_golden_fewshot.py

# few-shot 예시를 특정 논문으로 고정 (재현 가능한 비교를 위해 권장)
python3 ICR/compare_with_golden_fewshot.py --fewshot-ids 3,11,22

# 논문당 3회 호출해 다수결로 최종 코드 채택
python3 ICR/compare_with_golden_fewshot.py --fewshot-ids 3,11,22 --runs-per-paper 3

# 특정 논문만 / 무작위 표본만 테스트
python3 ICR/compare_with_golden_fewshot.py --fewshot-ids 3,11,22 --ids 8,12
python3 ICR/compare_with_golden_fewshot.py --fewshot-ids 3,11,22 --sample 15 --seed 42

# few-shot 없이(zero-shot) 비교하고 싶으면
python3 ICR/compare_with_golden_fewshot.py --fewshot-n 0
```

결과는 `comparison_results_fewshot.csv`(전체)와 `comparison_results_fewshot_mismatches.csv`
(정답과 어긋난 것만)에 저장되고, 터미널에 완전일치·정답포함·ERR판정 요약이 출력된다.
`focus` 컬럼에는 1차(division) 판단 내용(`has_community`/`role_relevant`/`env_relevant`/근거)이
그대로 찍혀서, 왜 그렇게 나왔는지 바로 확인할 수 있다.

**주의**: "정답 카테고리 포함(gold_covered)"은 예측이 정답 코드를 **전부** 포함해야 True다 —
정답 2개 중 1개만 맞혀도 이 기준으로는 오답으로 집계된다(부분점수 아님). 부분점수를 보고
싶으면 `repeat_eval.py`의 F1/precision/recall을 참고.

### 3. 안정성 검증 (같은 평가를 N번 반복)

이 모델은 `temperature=0`이어도 같은 논문·같은 프롬프트에 매번 다르게 답할 수 있다(백엔드
비결정성). 한 번 돌린 결과만으로 "좋아졌다/나빠졌다"를 판단하면 노이즈에 속기 쉬우므로,
같은 평가를 반복해서 평균·표준편차를 보는 스크립트를 따로 만들었다.

```bash
python3 ICR/repeat_eval.py --repeats 100 --fewshot-ids 3,11,22 --runs-per-paper 3
```

- 반복마다 논문별 원자료를 `repeat_eval_raw.csv`에 즉시 append+flush (중단돼도 안전)
- 반복별 요약은 `repeat_eval_summary.csv`
- 다 끝나면 반복 간 "합산 정답률"/"합산 포함율"/"평균 F1"의 평균·표준편차를 출력
- 중단 후 이어서 하려면 raw CSV에서 마지막 반복 번호를 확인하고 `--start-repeat`으로 지정

100회 기준 논문당 최대 3단계 × `runs-per-paper`(기본 3) 호출이 들어가서, API 호출이
수만 건 나올 수 있다 — 오래 걸리는 작업이니 백그라운드로 돌리는 걸 권장한다.

## 주요 설계 결정과 이유

- **초록만 입력**: 제목의 "온라인 커뮤니티" 언급에 낚여 `has_community`를 잘못 판단하는 문제를
  막기 위함 (`classify_paper_fewshot.build_abstract_only_text`).
- **1차/2차/3차 분리**: 10개를 한 번에 판단시키는 부담을 줄이고, "1차가 틀리면 2차가 못
  되돌리는" 이전 2단계(축 결정→코드판정) 구조의 문제를 완화하기 위함. 창/장 반사실 테스트는
  2차(role) 호출 안에서 직접 이뤄지므로 그 판단이 코드 채점과 분리되지 않는다.
- **few-shot을 대화 턴이 아니라 시스템 프롬프트 텍스트로**: 멀티턴 구조 자체가 이 모델에서
  불안정했기 때문.
- **다수결(과반수)로 최종 채택, 완전합의 강제 안 함**: 3단계로 쪼개고 나니 3번 반복 시 "3번
  다 완전히 똑같이 나와야 인정"이라는 기준을 맞추기가 훨씬 어려워져서(각 반복마다 최대 3번의
  개별 호출이 있으니 변동 요인이 3배), 다수결(2/3)로 완화했다.
- **quote 필드는 "원문 또는 요지"**: 텍스트에 없는 근거를 지어내는 환각을 막기 위한 그라운딩
  장치. 완전한 자유 판단(quote 없이 결론만)으로 바꾸면 이 환각 문제가 다시 열릴 위험이 있어
  유지 중.

## 알려진 한계

- **`ROLE_FIELD_PUBLICSPHERE` ↔ `ROLE_WINDOW_MISC` (그리고 `ROLE_WINDOW_POLARIZE`) 경계**:
  같은 논문에 대해 3회 호출이 서로 다른 코드를 대는 현상이 반복 확인됐다(예: 정치토론
  게시판을 다룬 논문에서 WINDOW_MISC/FIELD_PUBLICSPHERE/WINDOW_POLARIZE가 호출마다 바뀜).
  프롬프트 문구를 여러 번 다듬었지만 완전히 안정화되지는 않았다 — 모델의 판단력/일관성
  한계로 보인다.
- **제목 제거의 트레이드오프**: 제목에만 "온라인 브랜드 커뮤니티" 같은 구체적 맥락이 있고
  초록 본문은 더 일반화된 표현("커뮤니티 효과", "커뮤니티 태도")만 쓰는 논문의 경우, 제목을
  빼면서 `has_community` 판단이 오히려 더 어려워지는 사례가 있었다.
- **golden 데이터 자체의 노이즈**: 검증 과정에서 CSV의 오타(`ROLE_` 접두사 누락, `GOVERNACE`
  스펠링 오류)와, `focus`(사람이 미리 써둔 요약) 컬럼과 `class1`(최종 정답) 컬럼이 서로
  모순되는 사례(예: focus는 "사회자본 형성의 장"이라 써놓고 class1은 ERR)를 발견해 일부는
  고쳤다. 나머지 모순 사례는 모델이 그 애매함을 그대로 반영해 판단이 갈리는 것으로 보인다.

## 사용하지 않는 파일

`prompt_template_system.txt`는 이전(요약 없이 단일 호출로 10개를 한 번에 판단하던) 버전의
시스템 프롬프트다. 지금은 `prompt_template_division/role/env.txt` 3개로 대체되어 어떤
Python 파일에서도 더 이상 참조하지 않는다 — 참고용으로만 남겨둔 것이라 삭제해도 무방하다.
