"""
2단계(two-stage) 분류 파이프라인.

기존 classify_paper.py는 STEP 1(focus 요약)부터 STEP 4(11개 코드 판정)까지를 한 번의
API 호출 안에서 전부 수행한다. 그런데 id=11(장원영 팬덤 논문)로 반복 테스트한 결과,
같은 호출 안에서는 모델이 STEP 2~4를 진행할 때 STEP 1에서 쓴 focus 문장을 진짜 전제로
쓰지 않고, 입력에 계속 남아있는 원본 초록을 그때그때 다시 참조해 focus와 무관한 근거를
끌어오는 현상이 반복 확인됐다(quote가 focus에 없는 원본 초록 문장인데도 focus와 상충되지
않는 것처럼 나옴). 이건 이 논문 하나만의 문제가 아니라, focus를 판정 전제로 쓰려는 이
프로젝트의 설계 의도 자체가 단일 호출 구조에서는 강제되지 않는다는 구조적 문제라 다른
논문에서도 재발할 수 있다.

이 모듈은 그래서 두 번의 독립된 API 호출로 나누고, 프롬프트 자체도 파일 두 개로 분리했다:
    1차 호출 — prompt_template_stage1.txt + 논문 원문(제목/저자/초록) → focus 문장만 뽑는다.
    2차 호출 — prompt_template_stage2.txt + 1차에서 나온 focus 문장(원본 초록은 주지 않음)
               → STEP 2~4(11개 코드 판정)를 수행한다.
두 파일은 각자 독립적으로 수정할 수 있다 — 기존 prompt_template.txt(단일 호출용)는 그대로
남겨두고 건드리지 않는다. 2차 호출 시점엔 모델이 원본 초록으로 돌아가 focus에 없는 근거를
끌어올 방법이 물리적으로 없다 — quote는 focus 문장 안에 실제로 있는 표현이거나 null이어야 한다.

(참고: 실험해보니 이 분리가 "focus에 없는 원본 인용"은 확실히 막았지만, 대신 모델이
focus 문장 자체를 quote로 재활용하며 과신하는 새로운 패턴이 나타났다. 즉 이 모듈은
문제 하나를 없애지만 전체가 저절로 정답을 맞히는 건 아니며, 여전히 검증이 필요하다.)

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."

사용 예:
    python3 classify_paper_two_stage.py --paper-file paper.txt
    python3 classify_paper_two_stage.py --paper-file paper.txt --reasoning-effort medium
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from classify_paper import MODEL, NVIDIA_BASE_URL, extract_json
from compare_with_golden import DEFAULT_CSV, build_paper_text

TEMPLATE_STAGE1_PATH = Path(__file__).parent / "prompt_template_stage1.txt"
TEMPLATE_STAGE2_PATH = Path(__file__).parent / "prompt_template_stage2.txt"


def build_stage1_messages(paper_text: str) -> list[dict]:
    """1차 호출: prompt_template_stage1.txt + 논문 원문 → focus 문장만 뽑는다."""
    template = TEMPLATE_STAGE1_PATH.read_text(encoding="utf-8")
    instructions, _, trailer = template.partition("{{PAPER_TEXT}}")
    return [
        {"role": "system", "content": instructions.rstrip()},
        {"role": "user", "content": paper_text + trailer},
    ]


def build_stage2_messages(focus: str) -> list[dict]:
    """2차 호출: prompt_template_stage2.txt + focus 문장(원본 초록은 주지 않음) → STEP 2~4 수행."""
    template = TEMPLATE_STAGE2_PATH.read_text(encoding="utf-8")
    instructions = template.replace("{{FOCUS}}", focus)
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": focus},
    ]


def _call(client: OpenAI, model: str, messages: list[dict], temperature: float, reasoning_effort: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=8192,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": reasoning_effort},
    )
    return response.choices[0].message.content


def classify_paper_two_stage(
    paper_text: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
) -> dict:
    """두 번의 API 호출로 논문을 분류한다. 반환값은 extract_json()을 거친 최종 dict에
    stage1_focus_raw(1차 호출 원문 응답)를 추가로 담는다."""
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    stage1_raw = _call(client, model, build_stage1_messages(paper_text), temperature, reasoning_effort)
    stage1_obj = json.loads(stage1_raw)
    focus = stage1_obj.get("focus")
    if not focus:
        raise ValueError(f"1차 호출에서 focus를 얻지 못했습니다. 원문: {stage1_raw[:500]!r}")

    stage2_raw = _call(client, model, build_stage2_messages(focus), temperature, reasoning_effort)
    parsed = extract_json(stage2_raw)
    parsed["stage1_focus_raw"] = stage1_raw
    return parsed


def load_paper_text_by_id(csv_path: Path, paper_id: str) -> tuple[str, dict]:
    """golden CSV에서 id로 논문을 찾아 build_paper_text() 형태로 변환한다.
    반환값은 (paper_text, gold_row) — gold_row에서 class1/class2를 참고용으로 볼 수 있다."""
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["id"] == paper_id), None)
    if row is None:
        sys.exit(f"오류: id={paper_id} 논문을 golden CSV에서 찾을 수 없습니다.")
    return build_paper_text(row), row


def main():
    parser = argparse.ArgumentParser(description="2단계(focus 먼저 → 그 focus만으로 코드 판정) 분류")
    parser.add_argument("--id", help="golden CSV의 논문 id (--paper-file 대신 사용)")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="golden CSV 경로 (--id 사용 시)")
    parser.add_argument("--paper-file", help="논문 텍스트(제목/저자/초록) 파일 경로 (--id 대신 사용)")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--output", default=None, help="결과 JSON을 저장할 파일 경로 (선택)")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["low", "medium", "high"],
        help="모델의 reasoning_effort (기본 low, classify_paper.py와 동일한 기본값)",
    )
    args = parser.parse_args()

    if not args.id and not args.paper_file:
        sys.exit("오류: --id 또는 --paper-file 중 하나는 필요합니다.")

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

    gold_row = None
    if args.id:
        paper_text, gold_row = load_paper_text_by_id(Path(args.csv), args.id)
        print(f"분류 중: id={args.id} - {gold_row['title'][:40]}...", file=sys.stderr)
    else:
        paper_text = Path(args.paper_file).read_text(encoding="utf-8")

    try:
        parsed = classify_paper_two_stage(
            paper_text, MODEL, api_key, reasoning_effort=args.reasoning_effort
        )
    except (ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"오류: {exc}")

    if gold_row:
        parsed = {
            "id": gold_row["id"],
            "title": gold_row["title"],
            "class1_raw": gold_row.get("class1", ""),
            "class2_raw": gold_row.get("class2", ""),
            "model_response": parsed,
        }

    pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
    print(pretty)

    if args.output:
        Path(args.output).write_text(pretty, encoding="utf-8")
        print(f"\n결과가 {args.output} 에 저장되었습니다.", file=sys.stderr)


if __name__ == "__main__":
    main()
