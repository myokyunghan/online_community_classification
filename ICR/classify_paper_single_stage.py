"""
ICR/ 폴더의 stage1+stage2 판정 기준을 한 번의 API 호출로 합친 prompt_template_single_stage.txt로
논문을 분류하는 스크립트. classify_paper_two_stage.py와 FIELD/WINDOW/ENV 판별 기준·quote 무효
규칙은 동일하지만, 이 버전은 원본 논문 텍스트를 계속 볼 수 있어 quote를 원문 어디서든 가져올
수 있다 (2단계 버전은 quote를 stage1이 뽑은 focus 문장 안으로만 제한했다). 두 방식의 정확도를
직접 비교하기 위한 스크립트다.

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."

사용 예:
    python3 ICR/classify_paper_single_stage.py --id 1
    python3 ICR/classify_paper_single_stage.py --paper-file paper.txt
    python3 ICR/classify_paper_single_stage.py --id 1 --reasoning-effort medium
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

# 루트에도 동명의 classify_paper.py가 없으므로 순서 상관은 없지만, 다른 ICR 스크립트와
# 일관성을 위해 루트 경로는 뒤에 append한다.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from classify_paper import MODEL, NVIDIA_BASE_URL  # noqa: E402
from compare_with_golden import build_paper_text  # noqa: E402

from classify_paper_two_stage import robust_extract_json  # noqa: E402  (이 폴더의 버전 — 코드펜스/이스케이프 복구 재사용)

ICR_DIR = Path(__file__).parent
TEMPLATE_PATH = ICR_DIR / "prompt_template_single_stage.txt"
DEFAULT_CSV = ICR_DIR / "온라인_커뮤니티_연구지형_분류논문_30편.csv"


def build_messages(paper_text: str) -> list[dict]:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    instructions, _, trailer = template.partition("{{PAPER_TEXT}}")
    return [
        {"role": "system", "content": instructions.rstrip()},
        {"role": "user", "content": paper_text + trailer},
    ]


def classify_paper_single_stage(
    paper_text: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
) -> dict:
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(paper_text),
        temperature=temperature,
        max_tokens=8192,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": reasoning_effort},
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        raise ValueError(
            f"API 응답 content가 비었습니다 (finish_reason={choice.finish_reason!r}, usage={response.usage})"
        )
    try:
        return robust_extract_json(content)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{exc} | 응답 일부: {content[:500]!r}") from exc


def load_paper_text_by_id(csv_path: Path, paper_id: str) -> tuple[str, dict]:
    """ICR golden CSV에서 id로 논문을 찾아 build_paper_text() 형태로 변환한다.
    반환값은 (paper_text, gold_row) — gold_row에서 class1/class2를 참고용으로 볼 수 있다."""
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["id"] == paper_id), None)
    if row is None:
        sys.exit(f"오류: id={paper_id} 논문을 golden CSV에서 찾을 수 없습니다.")
    return build_paper_text(row), row


def main():
    parser = argparse.ArgumentParser(description="ICR 판정 기준을 단일 호출로 수행하는 분류 (2단계 방식과 비교용)")
    parser.add_argument("--id", help="ICR golden CSV의 논문 id (--paper-file 대신 사용)")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="golden CSV 경로 (--id 사용 시)")
    parser.add_argument("--paper-file", help="논문 텍스트(제목/저자/초록) 파일 경로 (--id 대신 사용)")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--output", default=None, help="결과 JSON을 저장할 파일 경로 (선택)")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["low", "medium", "high"],
        help="모델의 reasoning_effort (기본 low)",
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
        parsed = classify_paper_single_stage(
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
