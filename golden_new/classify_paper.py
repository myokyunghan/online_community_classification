"""
자체 서버(143.248.248.192, Ollama로 띄운 qwen3.8:27b-mlx)를 OpenAI 호환 API로
공통 상수/유틸. golden_new/ 폴더 전용 — NVIDIA_BASE_URL/MODEL을 여기서만 바꿔서
ICR/(NVIDIA API 사용)에는 영향이 없다. classify_paper_fewshot.py/compare_with_golden_fewshot.py/
repeat_eval.py가 이 파일의 MODEL/NVIDIA_BASE_URL/derive_role_env_from_audit을 가져다 쓴다.

vLLM 서버는 API 키 인증을 안 걸어놨으므로, --api-key/NVIDIA_API_KEY 미지정 시 더미 문자열을
그대로 쓴다 (openai 클라이언트가 api_key를 필수로 요구해서 빈 값은 안 됨).

build_messages/classify_paper/main(단일 호출 CLI, prompt_template.txt 기반)은 golden_new의
3단계(division/role/env) 파이프라인에서 직접 쓰이진 않지만, 루트 compare_with_golden.py가
모듈 최상단에서 `from classify_paper import MODEL, classify_paper, extract_json`을 하기
때문에 남겨둔다 — golden_new 스크립트를 직접 실행하면 이 파일이 "classify_paper" 모듈로
먼저 로드돼 루트 버전을 가리므로, classify_paper 함수 자체가 없으면 그 import가 깨진다.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

NVIDIA_BASE_URL = "http://143.248.248.192:11434/v1"  # 자체 Ollama 서버 (OpenAI 호환 엔드포인트)
DUMMY_API_KEY = "not-needed"  # Ollama 서버는 인증을 안 걸어놔서 아무 문자열이나 허용

MODEL = "qwen3.8:27b-mlx"
TEMPLATE_PATH = Path(__file__).parent / "prompt_template.txt"


def build_messages(paper_text: str, model: str) -> list[dict]:
    """분류 지침은 system 메시지로, 논문 내용은 user 메시지로 분리한다."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    instructions, _, trailer = template.partition("{{PAPER_TEXT}}")
    return [
        {"role": "system", "content": instructions.rstrip()},
        {"role": "user", "content": paper_text + trailer},
    ]


REQUIRED_KEYS = ("focus", "audit", "rationale")


def extract_json(text: str) -> dict:
    """모델 응답에서 첫 JSON 객체만 추출한다. 앞뒤에 부연 설명이 붙어도 무시한다."""
    start = text.find("{")
    if start == -1:
        raise ValueError("응답에서 JSON 객체를 찾을 수 없습니다.")
    obj, _ = json.JSONDecoder().raw_decode(text, start)

    missing = [key for key in REQUIRED_KEYS if key not in obj]
    if missing:
        raise ValueError(
            f"응답이 불완전합니다 (누락된 키: {missing}). 모델이 전체 출력 스키마를 완성하지 못하고 중간에 끊긴 것으로 보입니다."
        )
    if not isinstance(obj.get("audit"), list) or not obj["audit"]:
        raise ValueError("응답의 audit 배열이 비어있거나 형식이 올바르지 않습니다.")

    return derive_role_env_from_audit(obj)


def derive_role_env_from_audit(parsed: dict) -> dict:
    """role/env를 모델이 따로 조립하게 하지 않고, audit 11개 코드 전부의 확률(prob)을 직접 모아서
    만든다. 모델이 audit과 다르게 role/env를 조립하다 생기는 누락·불일치를 원천 차단한다.

    11개 코드는 개별 이진 판단이 아니라 서로 비교한 상대적 확률로 매겨지므로, prob>=0.5인
    코드를 "부여"로 취급한다. quote 없이 prob>=0.5인 항목은 환각 방지 규칙(R1) 위반이라 에러 처리한다.
    부여된 코드가 3개 이상이면 확률 상위 2개만 남기고, 잘려나간 코드는 parsed["_dropped_by_cap"]에
    기록해 이후 단계에서 정답이 잘렸는지 추적할 수 있게 한다."""
    granted = []
    for item in parsed.get("audit", []):
        code = item.get("code", "")
        prob = item.get("prob")
        if not isinstance(prob, (int, float)):
            raise ValueError(f"'{code}'에 prob이 없습니다. audit 11개 항목 전부에 prob이 있어야 합니다.")
        if prob >= 0.5:
            if not item.get("quote"):
                raise ValueError(f"'{code}'의 prob이 {prob}(0.5 이상)인데 quote(근거)가 없습니다.")
            granted.append({"code": code, "prob": prob, "evidence": item.get("quote")})

    ranked = sorted(granted, key=lambda item: item["prob"], reverse=True)
    top2, rest = ranked[:2], ranked[2:]

    parsed["role"] = [item for item in top2 if item["code"].startswith("ROLE_")]
    parsed["env"] = [item for item in top2 if item["code"].startswith("ENV_")]
    parsed["status"] = "OK" if top2 else "ERR"
    parsed["_dropped_by_cap"] = [item["code"] for item in rest]
    return parsed


def classify_paper(paper_text: str, model: str, api_key: str, temperature: float = 0.0) -> str:
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=build_messages(paper_text, model),
        temperature=temperature,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="자체 서버(vLLM, Qwen)로 논문 분류")
    parser.add_argument("--paper-file", required=True, help="논문 텍스트(제목/초록/방법/결론 등) 파일 경로")
    parser.add_argument("--api-key", default=None, help="자체 서버 API 키 (인증 불필요 — 미지정 시 더미 값 사용)")
    parser.add_argument("--output", default=None, help="결과 JSON을 저장할 파일 경로 (선택)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY") or DUMMY_API_KEY

    paper_text = Path(args.paper_file).read_text(encoding="utf-8")

    raw_output = classify_paper(paper_text, MODEL, api_key)

    try:
        parsed = extract_json(raw_output)
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, ValueError):
        print("경고: 모델 응답이 유효한 JSON이 아닙니다. 원문을 그대로 출력합니다.", file=sys.stderr)
        pretty = raw_output

    print(pretty)

    if args.output:
        Path(args.output).write_text(pretty, encoding="utf-8")
        print(f"\n결과가 {args.output} 에 저장되었습니다.", file=sys.stderr)


if __name__ == "__main__":
    main()
