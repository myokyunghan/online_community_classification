#!/usr/bin/env bash
# vLLM 서버가 prompt_template.txt로 실제 어떻게 응답하는지 curl로 바로 확인하는 스크립트.
# classify_paper.py의 build_messages()/MODEL/샘플링 상수를 그대로 재사용해서 payload를 만들기
# 때문에, 실제 파이프라인이 보내는 요청과 동일한 요청이 나간다.
#
# 사용법:
#   ./curl_test.sh                          # 내장 샘플 논문 + classify_paper.py 기본 temperature로 테스트
#   ./curl_test.sh paper.txt                # 파일의 논문 텍스트(제목/초록 등)로 테스트
#   TEMPERATURE=0.01 ./curl_test.sh         # temperature만 덮어쓰기 (재현성 확인용)
#   TEMPERATURE=0.01 ./curl_test.sh paper.txt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_URL="$(python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from classify_paper import NVIDIA_BASE_URL; print(NVIDIA_BASE_URL.rstrip('/') + '/chat/completions')")"

PAPER_FILE="${1:-}"
PAYLOAD_FILE="$(mktemp)"
CLEANUP_FILES=("$PAYLOAD_FILE")

if [[ -z "$PAPER_FILE" ]]; then
    PAPER_FILE="$(mktemp)"
    CLEANUP_FILES+=("$PAPER_FILE")
    cat > "$PAPER_FILE" <<'SAMPLE'
제목: (curl 테스트용 샘플)
초록: 온라인 커뮤니티에서의 익명성이 이용자의 발화 행태에 미치는 영향을 분석했다.
SAMPLE
fi

trap 'rm -f "${CLEANUP_FILES[@]}"' EXIT

TEMPERATURE="${TEMPERATURE:-}"

python3 -c "
import json, os, sys
sys.path.insert(0, '$SCRIPT_DIR')
from classify_paper import build_messages, MODEL, SAMPLING_TEMPERATURE, ENABLE_THINKING

temperature_override = os.environ.get('TEMPERATURE')
temperature = float(temperature_override) if temperature_override else SAMPLING_TEMPERATURE

paper_text = open('$PAPER_FILE', encoding='utf-8').read()
payload = {
    'model': MODEL,
    'messages': build_messages(paper_text, MODEL),
    'temperature': temperature,
    'max_tokens': 8192,
    'response_format': {'type': 'json_object'},
    'chat_template_kwargs': {'enable_thinking': ENABLE_THINKING},
}
open('$PAYLOAD_FILE', 'w', encoding='utf-8').write(json.dumps(payload, ensure_ascii=False))
"

echo "POST $SERVER_URL" >&2
echo "curl -sS '$SERVER_URL' -H 'Content-Type: application/json' -d @payload.json" >&2
echo >&2

curl -sS "$SERVER_URL" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_FILE" | jq .
