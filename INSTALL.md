# 설치 및 실행 안내

다른 컴퓨터에서 이 프로젝트의 논문 분류(`golden_new/classify_targets.py`)와 그래프
그리기(`target_new/visualize_results.py`)를 실행하는 방법입니다.

## 1. 준비물

| 항목 | 내용 |
|---|---|
| Python | **3.9 이상**. macOS 기본 `python3`(3.9.6)로 충분하며 pyenv는 필요 없습니다. |
| LLM 서버 | Ollama 서버에 `qwen3.8:27b-mlx` 모델이 올라가 있어야 합니다. 분류 스크립트가 이 서버에 요청을 보냅니다. |
| 한글 폰트 | 그래프용. macOS는 기본 폰트(AppleGothic)를 씁니다. 다른 OS는 [6. 문제 해결](#6-문제-해결)을 보세요. |

## 2. 코드와 데이터 받기

```bash
git clone https://github.com/myokyunghan/online_community_classification.git
cd online_community_classification
```

이미 받아 둔 저장소가 있으면 `git pull`로 최신 상태로 맞춥니다. 실행에 필요한 파일은 다음과
같습니다. git에 커밋되지 않은 파일이 있으면 직접 복사해야 합니다.

| 파일 | 용도 |
|---|---|
| `golden_new/prompt_template_division.txt`, `_role.txt`, `_env.txt` | 1~3단계 프롬프트 |
| `golden_new/온라인_커뮤니티_연구지형_분류논문_30편.csv` | few-shot 예시(id 3, 11, 22)를 뽑는 정답 데이터 |
| `target_new/KCI 사회과학.xlsx` | 분류 대상 논문 693편 |
| `target_new/classification_results_runs3.csv` | 3회 분류 결과. **이미 끝난 논문을 건너뛰려면 반드시 필요**합니다. 없으면 693편을 처음부터 다시 돌립니다. |
| `requirements-lock.txt` | 검증된 패키지 버전 목록 |

## 3. 가상환경과 패키지 설치

시스템 파이썬에 이미 깔린 패키지와 섞이지 않도록 프로젝트 전용 가상환경을 만듭니다.
`.venv/`는 `.gitignore`에 들어 있어 커밋되지 않습니다.

```bash
python3 -m venv .venv            # 처음 한 번만
source .venv/bin/activate        # 터미널을 새로 열 때마다
pip install --upgrade pip
pip install -r requirements-lock.txt
```

설치가 끝났는지 확인합니다. `ok`가 나오면 됩니다.

```bash
python -c "import pandas, openpyxl, openai, httpx, matplotlib; print('ok')"
```

- `requirements-lock.txt`는 Python 3.9.6에서 설치하고 분류와 그래프까지 실행해 본 버전 목록입니다.
  주요 패키지는 pandas 2.3.3, openai 2.48.0, openpyxl 3.1.5, httpx 0.28.1, matplotlib 3.9.4,
  numpy 2.0.2입니다. numpy와 matplotlib은 Python 3.9에서 설치할 수 있는 마지막 버전에 맞췄습니다.
- 루트의 `requirements.txt`는 예전 스크립트용으로 버전 하한만 적혀 있습니다. 설치에는
  `requirements-lock.txt`를 쓰세요.

## 4. 서버 주소 설정

서버 주소와 모델은 `golden_new/classify_paper.py` 위쪽의 상수에서 정합니다.

```python
NVIDIA_BASE_URL = "http://143.248.248.192:11434/v1"  # Ollama의 OpenAI 호환 주소
MODEL = "qwen3.8:27b-mlx"
```

- 같은 컴퓨터에서 Ollama를 띄우면 `http://localhost:11434/v1`로 바꿉니다.
- 서버에 모델이 있는지는 서버 쪽에서 `ollama list`로 확인합니다.
- `qwen3.8:27b-mlx`는 MLX 모델이라 Apple Silicon Mac에서만 돌아갑니다.

실행 전에 서버가 응답하는지 확인합니다. `http=200`이 나와야 합니다.

```bash
curl -s -m 60 -o /dev/null -w "http=%{http_code} %{time_total}s\n" \
  http://143.248.248.192:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8:27b-mlx","messages":[{"role":"user","content":"1"}],"max_tokens":3}'
```

## 5. 실행

모든 명령은 저장소 루트에서, 가상환경을 켠 상태(`(.venv)` 표시)로 실행합니다.

### 5-1. 논문 분류

먼저 2편만 돌려서 전체 흐름이 도는지 확인합니다. 결과는 임시 파일에 저장해 기존 결과를 건드리지 않습니다.

```bash
python golden_new/classify_targets.py --output /tmp/test.csv --limit 2 --runs-per-paper 1 --overwrite
```

이상이 없으면 본 실행을 합니다.

```bash
python golden_new/classify_targets.py \
  --output target_new/classification_results_runs3.csv --runs-per-paper 3
```

- `--runs-per-paper 3`: 논문마다 3회 호출하고, 2회 이상 나온 라벨만 채택합니다. 3회가 전부
  다르게 나오면 라벨이 빈칸으로 남으며, 이런 논문은 사람이 검토합니다.
- 결과는 논문 한 편이 끝날 때마다 저장됩니다. 중간에 끊겨도 **같은 명령을 다시 실행하면**
  끝난 논문은 건너뛰고 이어서 돌립니다.
- 시간은 논문 한 편에 3~8분 걸립니다. 요청을 한 번에 하나씩 순서대로 보냅니다.
- 터미널을 닫아도 계속 돌리려면 `nohup`이나 `tmux`를 씁니다.

```bash
nohup python golden_new/classify_targets.py \
  --output target_new/classification_results_runs3.csv --runs-per-paper 3 \
  > target_new/run_runs3.log 2>&1 &
tail -f target_new/run_runs3.log     # 진행 상황 보기
```

### 5-2. 실패(FAILED)한 논문 다시 돌리기

서버가 응답하지 않으면 해당 논문이 `status=FAILED`로 기록됩니다. FAILED 논문은 "끝난 논문"으로
치지 않으므로, 서버가 살아난 뒤 5-1의 본 실행 명령을 그대로 다시 실행하면 FAILED 논문만 다시
분류합니다.

다시 돌리면 결과 파일에 옛 FAILED 행과 새 결과 행이 함께 남습니다. 다시 돌리기 전에 FAILED 행을
지워 두면 파일이 깔끔합니다. 아래 명령은 백업을 만든 뒤 FAILED 행과 중복 행을 지웁니다.

```bash
cp target_new/classification_results_runs3.csv target_new/classification_results_runs3_backup.csv
python - <<'EOF'
import csv
p = "target_new/classification_results_runs3.csv"
with open(p, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
seen, keep = set(), []
for r in rows:
    if r["status"] == "FAILED" or r["id"] in seen:
        continue
    seen.add(r["id"])
    keep.append(r)
with open(p, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(keep)
print(f"{len(rows)}행 -> {len(keep)}행")
EOF
```

### 5-3. 그래프 그리기

```bash
python target_new/visualize_results.py
```

`target_new/charts/`에 PNG 15개와 PDF 2개(3·5 합본, 압축본)가 생깁니다. 읽는 결과 파일은
`visualize_results.py`의 `RESULTS_CSV`입니다. 현재는 1회 분류 결과(`classification_results.csv`)를
가리키므로, 3회 분류가 끝나면 `classification_results_runs3.csv`로 바꿔서 다시 그립니다.

## 6. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'six'` 등 import 에러 | 가상환경 없이 시스템 파이썬으로 실행했고, 거기 깔린 패키지가 깨져 있는 경우입니다. 3장대로 가상환경을 만들고 `source .venv/bin/activate` 후 실행하세요. |
| `TypeError: unsupported operand type(s) for \|: 'type' and 'NoneType'` | 예전 버전의 `compare_with_golden.py`입니다. Python 3.9는 파일 맨 위에 `from __future__ import annotations`가 있어야 합니다. `git pull`로 최신 파일을 받으세요. |
| `Request timed out` / `Connection error`가 연달아 나고 FAILED가 쌓임 | 서버가 멈춘 상태입니다. 4장의 `curl` 명령으로 확인하세요. 그동안의 경험상 서버가 약 10시간 연속으로 돌고 나면 응답하지 않는 일이 반복됐습니다. 이럴 때는 서버의 Ollama를 재시작한 뒤 5-2대로 다시 돌립니다. |
| 그래프의 한글이 네모(□)로 깨짐, 또는 "한글 폰트를 찾지 못해" 경고 | 한글 폰트가 없는 환경입니다(주로 Linux). 스크립트는 AppleGothic(macOS), 나눔고딕(Linux), 맑은 고딕(Windows) 중 설치된 것을 찾아 씁니다. Ubuntu는 `sudo apt install fonts-nanum` 후 matplotlib 폰트 캐시를 지우고(`rm -rf ~/.cache/matplotlib`) 다시 실행하세요. |
| `--concurrency`를 줘도 빨라지지 않음 | 현재 Ollama 서버는 요청을 한 번에 하나씩만 처리합니다. 병렬로 처리하려면 서버를 `OLLAMA_NUM_PARALLEL=2` 이상으로 띄워야 하며, MLX 모델에서 동작하는지는 확인하지 않았습니다. |
