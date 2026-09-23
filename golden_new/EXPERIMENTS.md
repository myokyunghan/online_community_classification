# 실험 실행 기록

옵션을 주고 실행하는 평가/테스트(`compare_with_golden_fewshot.py`, `repeat_eval.py` 등)는
실행할 때마다 이 표에 새 행(`ver#`)을 추가한다. 기존 행은 절대 수정하지 않고 append만 한다 —
나중에 "언제 뭘 어떤 설정으로 돌려서 어디에 저장했는지" 역추적하기 위한 기록이다.

새 항목을 추가할 때 채우는 것:
- **Ver**: 마지막 번호 다음 번호
- **일시**: 실행/완료 시각 (결과 파일 mtime 기준, 정확한 시작 시각을 모르면 "완료:"로 표기)
- **명령어**: 실제로 실행한 전체 명령어 (옵션 그대로)
- **설정**: 그 시점의 백엔드(vLLM/Ollama/로컬 mlx-lm)/모델/thinking on-off 등, 코드만 봐서는 알기 어려운 문맥
- **결과 파일**: 저장된 경로 (`golden_new/` 상대경로 또는 `/tmp/...`면 휘발성이라고 표시)
- **비고**: 핵심 결과 수치나 왜 중단/실패했는지 등

**파일명 규칙(ver13부터 적용)**: `--output`/`--raw-output`/`--summary-output`으로 저장하는 결과
파일명은 이 표의 Ver 번호를 그대로 따른다 — 설명적인 이름(`ollama_thinkoff_full27.csv` 같은) 대신
`verN.csv`(compare_with_golden_fewshot.py — mismatches는 자동으로 `verN_mismatches.csv`가 됨),
`verN_raw.csv`/`verN_summary.csv`(repeat_eval.py)로 저장한다. 무슨 설정으로 어떤 결과가 나왔는지는
파일명이 아니라 이 표를 봐야 알 수 있게 되므로, 실행할 때마다 반드시 이 표에 먼저(또는 바로
직후) 기록한다. ver12까지는 소급 적용하지 않고 기존 이름 그대로 둔다.

## 기록

| Ver | 일시 | 명령어 | 설정 | 결과 파일 | 비고 |
|-----|------|--------|------|-----------|------|
| 1 | 2026-09-18 완료 | `compare_with_golden_fewshot.py --sample 5 --seed 7 --fewshot-ids 3,11,22` | vLLM, Qwen3.8-27B-4bit, temperature=0.01 | `/tmp/golden_new_check5.csv` (휘발성, 소실됨) | 완전일치 3/5, 정답포함 4/5 |
| 2 | 2026-09-18 완료 | 위와 동일 + `--concurrency 5` | 동일 | `/tmp/golden_new_check5_concurrent.csv` (휘발성) | 결과 ver1과 동일(결정론적), 2분48초 소요 |
| 3 | 2026-09-18 완료 | 위와 동일 + `--concurrency 1` | 동일 | `/tmp/golden_new_check5_sequential.csv` (휘발성) | 결과 동일, 4분38초 소요 (concurrency 속도 비교용) |
| 4 | 2026-09-18 22:13 | `repeat_eval.py --repeats 20 --fewshot-ids 3,11,22 --runs-per-paper 3 --sample 10 --seed 7 --concurrency 5 --start-repeat 1 --raw-output repeat_eval_sample10_raw.csv --summary-output repeat_eval_sample10_summary.csv` | vLLM, Qwen3.8-27B-4bit, temperature=0.01 | `golden_new/repeat_eval_sample10_raw.csv`/`_summary.csv` | 중복 프로세스 문제로 초반에 중단 |
| 5 | 2026-09-19 00:53 | `repeat_eval.py --repeats 5 --fewshot-ids 3,11,22 --runs-per-paper 3 --sample 10 --seed 7 --concurrency 5 --start-repeat 1 --raw-output repeat_eval_qwen25_sample10_raw.csv --summary-output repeat_eval_qwen25_sample10_summary.csv` | vLLM, Qwen2.5-32B-Instruct-8bit로 모델 교체 직후 | `golden_new/repeat_eval_qwen25_sample10_raw.csv`/`_summary.csv` | seed 지정 방식에 대한 논의로 중단 (0바이트로 남음) |
| 6 | 2026-09-19 01:31 | `repeat_eval.py --repeats 5 --fewshot-ids 3,11,22 --runs-per-paper 3 --sample 10 --concurrency 5 --start-repeat 1 --raw-output repeat_eval_qwen25_sample10_noseed_raw.csv --summary-output repeat_eval_qwen25_sample10_noseed_summary.csv` | vLLM, Qwen2.5-32B-Instruct-8bit, temperature=0.01 (seed 없이) | `golden_new/repeat_eval_qwen25_sample10_noseed_raw.csv`/`_summary.csv` | 반복1·2 모두 합산정답률 60%/정답포함 70%/F1 0.667 (완전 재현) — id27 ERR오판, id15 축 오판 확인 |
| 7 | 2026-09-19 02:43 | `repeat_eval.py --repeats 5 --fewshot-ids 3,11,22 --runs-per-paper 3 --sample 5 --resample-each-repeat --concurrency 10 --start-repeat 1 --raw-output repeat_eval_qwen25_resample5_raw.csv --summary-output repeat_eval_qwen25_resample5_summary.csv` | vLLM, Qwen2.5-32B-Instruct-8bit, 샘플링 설정 전부 제거(서버 기본값) | `golden_new/repeat_eval_qwen25_resample5_raw.csv`/`_summary.csv` | 반복1: 합산정답률 40%/정답포함 60%/F1 0.667. Qwen2.5 공식 generation_config 기본값(temp=0.7/top_p=0.8/top_k=20/rep_penalty=1.05) 적용 위해 중단 |
| 8 | 2026-09-19 사용자 실행, 완료: 11:08(summary)/11:36(raw) | `repeat_eval.py --repeats 5 --fewshot-ids 3,11,22 --runs-per-paper 3 --sample 20 --resample-each-repeat --concurrency 10 --start-repeat 1 --csv golden_new/온라인_커뮤니티_연구지형_분류논문_30편.csv --raw-output golden_new/repeat_eval_qwen25_resample20_raw.csv --summary-output golden_new/repeat_eval_qwen25_resample20_summary.csv` | vLLM, Qwen2.5-32B-Instruct-8bit, temp=0.7/top_p=0.8/top_k=20/rep_penalty=1.05 | `golden_new/repeat_eval_qwen25_resample20_raw.csv`/`_summary.csv` | 결과가 계속 안 좋게 나와서 이후 Ollama로 서빙 스택 전환 논의로 이어짐 |
| 9 | 2026-09-19 00:52, 사용자 실행 | `compare_with_golden_fewshot.py --ids 13,6,15 --fewshot-ids 3,11,22 --output golden_new/comparison_remote_test3.csv` | vLLM, Qwen2.5-32B-Instruct-8bit | `golden_new/comparison_remote_test3.csv`/`_mismatches.csv` | id13/6/15 재현성 확인용 |
| 10 | 2026-09-19 완료 | `compare_with_golden_fewshot.py --sample 5 --seed 42 --fewshot-ids 3,11,22 --concurrency 5 --output /tmp/ollama_think_on_test5.csv` | **Ollama**(포트 11434)로 전환, qwen3.8:27b-mlx, thinking ON(OpenAI 호환 엔드포인트) | `/tmp/ollama_think_on_test5.csv` (휘발성) | 완전일치 2/5, 정답포함 2/5, JSON 파싱 에러 1건, 논문당 1~2분 |
| 11 | 2026-09-19 완료 | `compare_with_golden_fewshot.py --sample 5 --seed 42 --fewshot-ids 3,11,22 --concurrency 5 --output /tmp/ollama_think_off_test5.csv` | Ollama, qwen3.8:27b-mlx, thinking OFF(네이티브 `/api/chat`, `USE_OLLAMA_NATIVE_THINK_OFF=True`) | `/tmp/ollama_think_off_test5.csv` (휘발성) | 완전일치 2/5, 정답포함 **3/5**, 파싱 에러 없음, 논문당 20~25초 — think OFF가 우세해서 이걸로 확정 |
| 12 | 2026-09-19 12:23, 중단됨 | `compare_with_golden_fewshot.py --fewshot-ids 3,11,22 --concurrency 5 --output ollama_thinkoff_full27.csv` | Ollama, qwen3.8:27b-mlx, thinking OFF(네이티브) | `golden_new/ollama_thinkoff_full27.csv` | concurrency가 Ollama에서 의미 있는지 확인 안 된 채 진행하다 사용자 요청으로 중단 (1편만 기록됨) |
| 13 | 2026-09-19 완료 | `compare_with_golden_fewshot.py --fewshot-ids 3,11,22 --output ver13.csv` | Ollama, qwen3.8:27b-mlx, thinking ON(OpenAI 호환 엔드포인트, `USE_OLLAMA_NATIVE_THINK_OFF=False`), concurrency 없이 순차 | `golden_new/ver13.csv`/`_mismatches.csv` | **완전일치 23/27(85%), 정답포함 24/27(89%), 완전합의 27/27**. 5편 소표본(ver10)에서 안 좋게 나왔던 것과 달리 27편 전체에서는 지금까지 중 최고 수치 — 소표본 결과가 대표성이 없었던 것으로 보임. 어긋난 4편: id8(부분일치), id19·id20(ROLE 세부코드/ERR 경계), id27(상습 오판 — 커뮤니티 무관으로 오판). 소요시간 미측정(체감상 1시간+, 30분 모니터 창 2번 넘김) |
| 14 | 2026-09-19 완료 | `compare_with_golden_fewshot.py --fewshot-ids 3,11,22 --output ver14.csv` (`time`로 소요시간 측정, `2>ver14_time.log`) | vLLM(포트 11435)으로 복귀, mlx-community/Qwen3.8-27B-4bit, thinking ON(`ENABLE_THINKING=True`, chat_template_kwargs), top_p/top_k 미지정(vLLM 기본값), concurrency 없이 순차 | `golden_new/ver14.csv`/`_mismatches.csv`, 소요시간 로그 `golden_new/ver14_time.log` | **완전일치 18/27(67%), 정답포함 20/27(74%), 완전합의 27/27, 소요시간 14분44초**. ver13(Ollama)과 정확도 격차 큼(67% vs 85%) — 이후 원인 조사 결과 `ollama show qwen3.8:27b-mlx`에서 Modelfile 기본값이 `top_p=0.95/top_k=20`으로 박혀있는 걸 발견(ver13은 이 값이 암묵적으로 적용됐고, vLLM인 ver14는 top_p/top_k를 안 보내서 사실상 무제한 샘플링이었을 것으로 추정) → ver15에서 맞춰서 재검증 |
| 15 | 2026-09-19 완료 | `compare_with_golden_fewshot.py --fewshot-ids 3,11,22 --output ver15.csv` (`time`로 소요시간 측정, `2>ver15_time.log`) | vLLM, mlx-community/Qwen3.8-27B-4bit, thinking ON, **top_p=0.95/top_k=20을 Ollama Modelfile과 동일하게 명시적으로 추가**(`SAMPLING_TOP_P`/`SAMPLING_TOP_K` 신설, `classify_paper.py`/`classify_paper_fewshot.py` 양쪽 다 적용), concurrency 없이 순차 | `golden_new/ver15.csv`/`_mismatches.csv`, 소요시간 로그 `golden_new/ver15_time.log` | **ver14와 결과가 바이트 단위로 완전히 동일**(완전일치 18/27, 정답포함 20/27, 소요시간 14분44초로 거의 같음, `diff`로 확인) — top_p/top_k를 Ollama Modelfile 기본값과 맞춰도 vLLM 결과에 전혀 영향 없었음. **"Modelfile 기본값 때문" 가설은 기각**. ver13(Ollama, 85%)과 ver14/15(vLLM, 67%)의 격차 원인은 top_p/top_k가 아니며, 남은 유력 후보는 모델 양자화 방식 차이(vLLM: mlx-community MLX 4-bit vs Ollama: nvfp4) — 같은 "Qwen3.8-27B"라는 이름이지만 실제로는 다른 가중치일 가능성이 큼. 다음 검증하려면 같은 양자화로 두 서버에 동일 가중치를 올려야 함. (추가 확인: Ollama가 Modelfile 기본 temperature=1 대신 우리가 보낸 temperature=0.7을 실제로 반영하는지도 별도로 검증함 — temp=0.01일 때 5회 완전 동일, temp=2.0일 때 5회 다르게 나와서 override가 정상 작동함을 확인. temperature도 격차 원인에서 제외). **가중치 용량 비교로 확정**: HF `mlx-community/Qwen3.8-27B-4bit` 실제 파일 총합 16.08GB vs Ollama `qwen3.8:27b-mlx` `/api/tags` 기준 20.47GB — 같은 27.8B 모델인데 27% 용량 차이. "4-bit"라는 이름은 같지만 실제 양자화 스킴(그룹 크기 등)이 달라 정밀도가 다른 것으로 결론 — Ollama 쪽이 더 정밀한(더 큰) 양자화라 정확도는 높고 속도는 느린 트레이드오프로 설명됨 |
| 16 | 2026-09-19 완료 | `compare_with_golden_fewshot.py --fewshot-ids 3,11,22 --output ver16.csv` (`time`로 소요시간 측정, `2>ver16_time.log`) | vLLM, **mlx-community/Qwen3-30B-A3B-4bit로 모델 교체**(MoE, 총 30B/활성 3B), ver14와 동일 옵션(thinking ON, top_p/top_k 미지정), concurrency 없이 순차 | `golden_new/ver16.csv`/`_mismatches.csv`, 소요시간 로그 `golden_new/ver16_time.log` | **완전일치 11/27(41%), 정답포함 14/27(52%), 소요시간 7분47초**. 지금까지 중 최저 정확도 — "더 큰 모델이 더 낫다" 가설 기각. MoE는 총 파라미터(30B)는 많지만 토큰당 실제 연산에 쓰이는 활성 파라미터는 3B뿐이라, 이 분류 작업처럼 미묘한 구분이 필요한 추론에는 27B dense 모델(토큰당 27B 전부 사용)보다 실질적으로 얕은 추론이 된 것으로 추정. 속도는 ver14/15(14분44초) 대비 거의 2배 빠름 — 정확도-속도 트레이드오프. (참고: vLLM 서버를 이 모델로 재시작하는 과정에서 `--port`/`--host`/`--enforce-eager` 옵션이 줄바꿈 때문에 파싱 안 돼 기본 포트(8000, 방화벽 미개방)로 떴던 문제를 발견·수정함) |

<!-- 다음 행부터 이어서 추가 -->

## 최종 결정 (2026-09-19)

ver1~16 테스트 결과, **ver13 설정(Ollama, qwen3.8:27b-mlx, thinking ON)을 최종 확정**한다 —
완전일치 85%로 전 버전 중 최고. vLLM(affine 4bit, 67%)과의 격차는 서빙 스택이 아니라
양자화 정밀도 차이(Ollama: nvfp4 group_size=16 vs vLLM: affine group_size=64)로 결론.
vLLM 쪽에 더 정밀한 양자화(nvfp4/8bit)를 올리는 시도는 이 서버의 vLLM 환경(vllm_metal
플러그인)이 이 VLM 아키텍처(Qwen3_5, image-text-to-text)의 이미지 프로세서 로딩 단계에서
버그가 있어 실패함(`--trust-remote-code`로도 안 고쳐짐) — 추가 조사는 보류.

**운용 방침**: 프롬프트 수정·빠른 확인·반복 실험은 vLLM(`mlx-community/Qwen3.8-27B-4bit`)으로,
최종 확정 수치는 Ollama(`qwen3.8:27b-mlx`)로 몇 회 반복해서 검증. `classify_paper.py`의
`NVIDIA_BASE_URL`/`MODEL` 주석 처리된 줄들을 바꿔가며 전환한다.
