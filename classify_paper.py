"""
NVIDIA NIM API(build.nvidia.com)를 통해 논문을 분류 프롬프트로 분석하는 스크립트.

사전 준비:
    1. https://build.nvidia.com 에서 무료 API 키 발급
    2. `export NVIDIA_API_KEY="nvapi-..."` 로 환경변수 설정
    3. `pip install openai` (NVIDIA NIM은 OpenAI 호환 API를 제공)

사용 예:
    python classify_paper.py --paper-file paper.txt
    python classify_paper.py --paper-file paper.txt --output result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# MODEL = "qwen/qwen3-next-80b-a3b-instruct"  # 2026-07-27 end of life로 사용 불가 (Qwen 계열 전체가 카탈로그에서 빠짐)
# MODEL = "meta/llama-3.3-70b-instruct"  # dense 70B라 논문 693편 처리하기엔 너무 느림
# MODEL = "nvidia/nemotron-3-nano-30b-a3b"  # 속도는 빠르나 FIELD/WINDOW 축을 자주 혼동해 golden 정확도가 낮음(2/9)
# MODEL = "nvidia/nemotron-3-super-120b-a12b"  # thinking on/off 모두 스키마 준수가 불안정(사고 누출/audit이 객체로 나옴/필드 누락/딴 스키마 등 네 가지 실패 유형)
# MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"  # /v1/models엔 있으나 이 계정에 실제 호출 권한 없음 (404 Function Not Found)
# MODEL = "moonshotai/kimi-k2.6"
# MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"  # PUBLICSPHERE/MISC 혼동이 nano 모델과 동일하게 재발 (id=12,8,18) -- 프롬프트 경계사례 의심
# MODEL = "nvidia/nemotron-3-ultra-550b-a55b"  # Nemotron-3 계열 전체가 크기 무관하게 요청 스키마를 무시하고 자체 스키마를 생성함
# MODEL = "mistralai/mistral-medium-3.5-128b"
MODEL = "openai/gpt-oss-120b"  # harmony 포맷 추론모델. reasoning_effort로 사고 비중 조절, structured output 네이티브 지원
TEMPLATE_PATH = Path(__file__).parent / "prompt_template.txt"

# 이 모델은 reasoning on/off를 chat_template_kwargs가 아니라 system 메시지 내용으로 제어하며,
# 문서상 "모든 지시사항은 user 프롬프트에 담아야 한다"고 명시돼 있어 system/user 분리 방식이 다르다.
NEMOTRON_SUPER_LLAMA_MODELS = {"nvidia/llama-3.3-nemotron-super-49b-v1.5"}


def build_messages(paper_text: str, model: str) -> list[dict]:
    """분류 지침은 system 메시지로, 논문 내용은 user 메시지로 분리한다.

    단, Llama-3.3-Nemotron-Super 계열은 system 메시지를 reasoning on/off 스위치로만 쓰고
    지시사항은 전부 user 메시지에 담아야 하므로 예외적으로 다르게 구성한다."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    instructions, _, trailer = template.partition("{{PAPER_TEXT}}")

    if model in NEMOTRON_SUPER_LLAMA_MODELS:
        return [
            {"role": "system", "content": "detailed thinking off"},
            {"role": "user", "content": instructions.rstrip() + "\n\n" + paper_text + trailer},
        ]

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
        extra_body={"reasoning_effort": "low"},
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="NVIDIA NIM API로 논문 분류")
    parser.add_argument("--paper-file", required=True, help="논문 텍스트(제목/초록/방법/결론 등) 파일 경로")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--output", default=None, help="결과 JSON을 저장할 파일 경로 (선택)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

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
