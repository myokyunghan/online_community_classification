"""
2단계 분리 실험: STEP 1(focus)은 이미 나왔다고 가정하고, 그 focus 문장 하나만
"유일한 입력 텍스트"로 주고 STEP 2~4(코드 판정)를 시켜본다. 원본 초록은 아예 주지 않는다.

목적: SOCIALCAPITAL이 "위계와 규범적 기대를 형성하였다" 같은 quote를 계속 재사용하는 게
"원본 초록을 언제든 다시 참조할 수 있어서"인지 확인 — focus 한 문장만 주면 그 quote 자체를
지어낼 방법이 없어야 정상이다(입력에 없는 문장이므로).

사용 예:
    python3 test_focus_only.py --id 11
    python3 test_focus_only.py --focus "이 논문은 온라인 커뮤니티를 ..."
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from classify_paper import MODEL, NVIDIA_BASE_URL, TEMPLATE_PATH, extract_json

DEFAULT_CSV = Path(__file__).parent / "golden" / "온라인_커뮤니티_연구지형_분류논문_22편.csv"


def build_focus_only_prompt(focus: str) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    step2_onward = "## STEP 2" + template.split("## STEP 2", 1)[1]
    step2_onward = step2_onward.replace("{{PAPER_TEXT}}", "").rstrip()

    return f"""당신은 온라인 커뮤니티 관련 논문의 연구지형을 분류하는 문헌고찰 연구자다.

이 논문에 대한 초점 요약은 이미 STEP 1에서 확정되었다 — 아래 한 문장이 지금 네가 볼 수 있는
**유일한 입력 텍스트**다. 원본 논문 전체(제목·저자·초록)는 너에게 주어지지 않는다.

> {focus}

이 문장에 실제로 없는 내용을 지어내 quote로 쓰지 않는다 — 이 문장이 뒷받침하지 못하는 코드는
quote를 null로 두고 낮은 prob을 준다. 이 문장을 기준으로 아래 STEP 2 → 3 → 4를 수행한다.

{step2_onward}"""


def main():
    parser = argparse.ArgumentParser(description="focus 문장만으로 STEP 2~4를 수행하는 2단계 실험")
    parser.add_argument("--id", help="golden CSV의 논문 id (해당 id의 class1/class2를 정답으로 함께 출력)")
    parser.add_argument("--focus", help="직접 입력할 focus 문장 (--id 대신 사용 가능)")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

    gold = None
    if args.focus:
        focus = args.focus
    elif args.id:
        with DEFAULT_CSV.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        row = next((r for r in rows if r["id"] == args.id), None)
        if row is None:
            sys.exit(f"오류: id={args.id} 논문을 찾을 수 없습니다.")
        gold = {"class1": row.get("class1", ""), "class2": row.get("class2", "")}
        focus = input(f"id={args.id} ({row['title'][:40]}...)의 focus 문장을 붙여넣으세요: ").strip()
    else:
        sys.exit("오류: --id 또는 --focus 중 하나는 필요합니다.")

    system_prompt = build_focus_only_prompt(focus)

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": focus},
        ],
        temperature=0.0,
        max_tokens=8192,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "low"},
    )
    raw = response.choices[0].message.content

    try:
        parsed = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"파싱 실패: {exc}\n원문: {raw[:1000]}")

    if gold:
        print(f"(참고) 정답: class1={gold['class1']} class2={gold['class2']}\n", file=sys.stderr)

    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
