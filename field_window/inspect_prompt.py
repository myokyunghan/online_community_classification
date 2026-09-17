"""
classify_paper_fewshot.py가 실제 API 호출 시 만드는 system/user 메시지를 그대로 화면에
출력한다 (기본은 호출 없이 프롬프트만 확인 — --call을 주면 실제로 호출도 해서 응답까지 본다).

1차(division)는 few-shot이 없고, 2차(role)/3차(env)는 --fewshot-ids로 지정한 논문들이
few-shot 블록으로 시스템 프롬프트 뒤에 어떻게 붙는지까지 그대로 보여준다.

사용 예:
    python3 field_window/inspect_prompt.py --id 1
    python3 field_window/inspect_prompt.py --id 1 --fewshot-ids 3,11,22
    python3 field_window/inspect_prompt.py --id 1 --stage role
    python3 field_window/inspect_prompt.py --id 1 --call            # 실제 API 호출까지
    python3 field_window/inspect_prompt.py --paper-file paper.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classify_paper import MODEL, NVIDIA_BASE_URL  # noqa: E402
from classify_paper_fewshot import (  # noqa: E402  (이 폴더의 버전)
    _call,
    build_abstract_only_text,
    build_division_messages,
    build_env_messages,
    build_ideal_env_output,
    build_ideal_role_output,
    build_role_messages,
    robust_extract_division_json,
    robust_extract_raw_audit,
    robust_extract_role_choice,
)
from compare_with_golden import load_rows  # noqa: E402

DEFAULT_CSV = Path(__file__).parent / "온라인_커뮤니티_연구지형_분류논문_30편.csv"

SEP = "=" * 78


def print_messages(label: str, messages: list[dict]) -> None:
    print(f"\n{SEP}\n[{label}] 실제로 보내는 messages\n{SEP}")
    for msg in messages:
        content = msg["content"]
        print(f"\n--- role: {msg['role']} ({len(content)}자) ---\n")
        print(content)


def build_fewshot_pools(csv_path: Path, fewshot_ids: str) -> tuple[list, list]:
    if not fewshot_ids:
        return [], []
    rows = load_rows(csv_path)
    id_to_row = {r["id"]: r for r in rows}
    wanted_ids = [s.strip() for s in fewshot_ids.split(",") if s.strip()]
    missing = [i for i in wanted_ids if i not in id_to_row]
    if missing:
        sys.exit(f"오류: few-shot id를 golden CSV에서 못 찾음: {', '.join(missing)}")
    fewshot_pool = [id_to_row[i] for i in wanted_ids]
    fewshot_examples = [(build_abstract_only_text(r), build_ideal_role_output(r)) for r in fewshot_pool]
    env_fewshot_examples = [(build_abstract_only_text(r), build_ideal_env_output(r)) for r in fewshot_pool]
    return fewshot_examples, env_fewshot_examples


def main():
    parser = argparse.ArgumentParser(description="1~3차 호출에 실제로 들어가는 프롬프트를 그대로 확인")
    parser.add_argument("--id", help="golden CSV의 논문 id (--paper-file 대신 사용)")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="golden CSV 경로 (--id 사용 시)")
    parser.add_argument("--paper-file", help="논문 텍스트(초록) 파일 경로 (--id 대신 사용)")
    parser.add_argument(
        "--fewshot-ids", default="3,11,22", help="쉼표로 구분한 few-shot 예시 id (기본 3,11,22, 빈 문자열이면 few-shot 없음)"
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["division", "role", "env", "all"],
        help="어느 단계의 프롬프트를 볼지 (기본 all — 1~3차 전부)",
    )
    parser.add_argument("--call", action="store_true", help="프롬프트만 보지 않고 실제 API 호출까지 해서 응답도 본다")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (--call 사용 시, 미지정 시 NVIDIA_API_KEY 환경변수)")
    parser.add_argument(
        "--reasoning-effort", default="low", choices=["low", "medium", "high"], help="--call 사용 시 reasoning_effort"
    )
    args = parser.parse_args()

    if not args.id and not args.paper_file:
        sys.exit("오류: --id 또는 --paper-file 중 하나는 필요합니다.")

    if args.id:
        rows = load_rows(Path(args.csv))
        row = next((r for r in rows if r["id"] == args.id), None)
        if row is None:
            sys.exit(f"오류: id={args.id} 논문을 golden CSV에서 찾을 수 없습니다.")
        paper_text = build_abstract_only_text(row)
        print(f"대상 논문: id={args.id} - {row['title'][:50]}... (gold class1={row.get('class1','')!r})", file=sys.stderr)
    else:
        paper_text = Path(args.paper_file).read_text(encoding="utf-8").strip()

    fewshot_examples, env_fewshot_examples = build_fewshot_pools(Path(args.csv), args.fewshot_ids)
    if args.fewshot_ids:
        print(f"few-shot 고정 id: {args.fewshot_ids}", file=sys.stderr)

    api_key = None
    client = None
    if args.call:
        api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            sys.exit("오류: --call 사용 시 NVIDIA API 키가 필요합니다 (--api-key 또는 NVIDIA_API_KEY 환경변수).")
        from openai import OpenAI

        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    stages = ["division", "role", "env"] if args.stage == "all" else [args.stage]

    division = None
    if "division" in stages:
        messages = build_division_messages(paper_text)
        print_messages("1차 division", messages)
        if args.call:
            raw = _call(client, MODEL, messages, 0.0, args.reasoning_effort)
            division = robust_extract_division_json(raw)
            print(f"\n>>> 실제 응답:\n{json.dumps(division, ensure_ascii=False, indent=2)}")

    if "role" in stages:
        messages = build_role_messages(paper_text, fewshot_examples)
        print_messages("2차 role", messages)
        if args.call:
            raw = _call(client, MODEL, messages, 0.0, args.reasoning_effort)
            role_obj = robust_extract_role_choice(raw)
            print(f"\n>>> 실제 응답:\n{json.dumps(role_obj, ensure_ascii=False, indent=2)}")

    if "env" in stages:
        messages = build_env_messages(paper_text, env_fewshot_examples)
        print_messages("3차 env", messages)
        if args.call:
            raw = _call(client, MODEL, messages, 0.0, args.reasoning_effort)
            env_obj = robust_extract_raw_audit(raw)
            print(f"\n>>> 실제 응답:\n{json.dumps(env_obj, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
