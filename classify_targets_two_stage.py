"""
target/ 폴더의 실제 분류 대상 논문을 2단계(focus 먼저 → 그 focus만으로 코드 판정) 파이프라인으로
분류한다. classify_targets.py와 로직·출력 형식은 동일하지만, 내부적으로
classify_paper()(단일 호출) 대신 classify_paper_two_stage()(1차: focus만, 2차: focus만으로
판정)를 사용한다. 두 파이프라인 중 어느 쪽이 실제로 더 정확한지는 golden 22편으로는 노이즈
때문에 확신 있게 가리기 어려웠으므로, 693편 규모에서 결과를 비교해보기 위한 스크립트다.

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."
    pip install -r requirements.txt   # pandas, openpyxl 포함

사용 예:
    python classify_targets_two_stage.py
    python classify_targets_two_stage.py --limit 5
    python classify_targets_two_stage.py --reasoning-effort medium
    python classify_targets_two_stage.py --overwrite
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from classify_paper import MODEL, NVIDIA_BASE_URL, extract_json
from classify_paper_two_stage import build_stage1_messages, build_stage2_messages, _call
from classify_targets import DEFAULT_XLSX, failed_result, load_processed_ids, load_target_rows, no_abstract_result
from compare_with_golden import aggregate_predictions, build_paper_text
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

DEFAULT_OUTPUT = Path(__file__).parent / "target" / "classification_results_two_stage.csv"

RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, TimeoutError, RateLimitError, InternalServerError)
PARSE_RETRY_WAIT_SECONDS = 2.0

FIELDNAMES = [
    "id",
    "title",
    "author",
    "published_year",
    "journal",
    "abstract_lang",
    "status",
    "role_codes",
    "env_codes",
    "dropped_by_cap",
    "focus",
    "rationale",
    "full_agreement",
    "error_runs",
    "run_details",
]


def classify_retry_kind(exc: Exception) -> str | None:
    if isinstance(exc, RETRYABLE_ERRORS):
        return "server"
    if isinstance(exc, BadRequestError) and "DEGRADED" in str(exc):
        return "server"
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "parse"
    return None


def classify_row_two_stage(row: dict, api_key: str, reasoning_effort: str, temperature: float = 0.0) -> dict:
    """1차(focus) + 2차(focus만으로 코드 판정) 호출을 수행하고, compare_with_golden의
    aggregate_predictions()가 기대하는 형태(predicted_codes/status/parsed/dropped_by_cap)로 맞춘다."""
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    paper_text = build_paper_text(row)

    stage1_raw = _call(client, MODEL, build_stage1_messages(paper_text), temperature, reasoning_effort)
    stage1_obj = json.loads(stage1_raw)
    focus = stage1_obj.get("focus")
    if not focus:
        raise ValueError(f"1차 호출에서 focus를 얻지 못했습니다. 원문: {stage1_raw[:500]!r}")

    stage2_raw = _call(client, MODEL, build_stage2_messages(focus), temperature, reasoning_effort)
    parsed = extract_json(stage2_raw)

    role_codes = [item["code"] for item in parsed.get("role", [])]
    env_codes = [item["code"] for item in parsed.get("env", [])]

    return {
        "parsed": parsed,
        "status": parsed.get("status", "OK"),
        "focus": parsed.get("focus"),
        "predicted_codes": set(role_codes) | set(env_codes),
        "dropped_by_cap": set(parsed.get("_dropped_by_cap") or []),
    }


def classify_row_with_retry_two_stage(
    row: dict, api_key: str, reasoning_effort: str, max_retries: int, retry_wait_seconds: float
) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            return classify_row_two_stage(row, api_key, reasoning_effort)
        except Exception as exc:
            kind = classify_retry_kind(exc)
            if kind is None or attempt >= max_retries:
                raise
            wait = retry_wait_seconds if kind == "server" else PARSE_RETRY_WAIT_SECONDS
            label = f"{retry_wait_seconds / 60:.1f}분" if kind == "server" else f"{PARSE_RETRY_WAIT_SECONDS:.0f}초"
            print(f"    일시적 오류 (시도 {attempt}/{max_retries}) - {label} 후 재시도: {exc}", file=sys.stderr)
            time.sleep(wait)


def classify_row_multi_two_stage(
    row: dict,
    api_key: str,
    reasoning_effort: str,
    runs: int,
    max_retries: int,
    retry_wait_seconds: float,
    sleep_seconds: float,
) -> list[dict]:
    predictions = []
    for run in range(1, runs + 1):
        predictions.append(
            classify_row_with_retry_two_stage(row, api_key, reasoning_effort, max_retries, retry_wait_seconds)
        )
        if run < runs:
            time.sleep(sleep_seconds)
    return predictions


def format_result(row: dict, predictions: list[dict], aggregate: dict) -> dict:
    role_codes = sorted(c for c in aggregate["majority_codes"] if c.startswith("ROLE_"))
    env_codes = sorted(c for c in aggregate["majority_codes"] if c.startswith("ENV_"))

    run_details = "; ".join(
        f"run{i}:{','.join(sorted(codes)) if codes else '없음'}"
        for i, codes in enumerate(aggregate["run_code_sets"], 1)
    )
    dropped_runs = [p.get("dropped_by_cap") or set() for p in predictions]
    dropped_by_cap = "; ".join(
        f"run{i}:{','.join(sorted(d)) if d else '없음'}" for i, d in enumerate(dropped_runs, 1)
    )
    focus = " | ".join(str(p.get("focus") or "") for p in predictions)
    rationale = " | ".join(str(p["parsed"].get("rationale", "")) for p in predictions)
    error_runs = f"{aggregate['err_count']}/{len(predictions)}"

    return {
        "id": row["id"],
        "title": row["title"],
        "author": row["author"],
        "published_year": row["published_year"],
        "journal": row["journal"],
        "abstract_lang": row["abstract_lang"],
        "status": aggregate["majority_status"],
        "role_codes": ", ".join(role_codes),
        "env_codes": ", ".join(env_codes),
        "dropped_by_cap": dropped_by_cap,
        "focus": focus,
        "rationale": rationale,
        "full_agreement": aggregate["full_agreement"],
        "error_runs": error_runs,
        "run_details": run_details,
    }


def main():
    parser = argparse.ArgumentParser(description="target/ 폴더 실제 논문을 2단계 파이프라인으로 분류")
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="분류 대상 엑셀 경로")
    parser.add_argument("--sheet", default="Sheet1", help="엑셀 시트 이름")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="결과를 저장할 CSV 경로")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--limit", type=int, default=None, help="테스트용으로 앞 N편만 처리")
    parser.add_argument("--start", type=int, default=0, help="이 순번부터 처리 (0-based, 기본 0)")
    parser.add_argument("--runs-per-paper", type=int, default=1, help="논문당 반복 호출 횟수, 과반수로 채택 (기본 1)")
    parser.add_argument(
        "--reasoning-effort", default="low", choices=["low", "medium", "high"], help="모델의 reasoning_effort (기본 low)"
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="API 호출 사이 대기 시간(초, 기본 1.0)")
    parser.add_argument("--max-retries", type=int, default=3, help="일시적 오류/파싱 실패 시 최대 재시도 횟수 (기본 3)")
    parser.add_argument("--retry-wait-minutes", type=float, default=3.0, help="서버 오류 재시도 전 대기 시간(분, 기본 3.0)")
    parser.add_argument(
        "--overwrite", action="store_true", help="기존 --output 파일을 무시하고 처음부터 다시 처리"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

    rows = load_target_rows(Path(args.xlsx), args.sheet)
    rows = rows[args.start :]
    if args.limit:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    processed_ids = set() if args.overwrite else load_processed_ids(output_path)
    if processed_ids:
        print(f"이미 처리된 {len(processed_ids)}편은 건너뜁니다 (--overwrite로 재처리 가능).", file=sys.stderr)

    write_header = args.overwrite or not output_path.exists()
    mode = "w" if args.overwrite else "a"
    out_f = output_path.open(mode, encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    todo = [r for r in rows if r["id"] not in processed_ids]
    print(f"처리 대상: {len(todo)}편 (전체 {len(rows)}편 중)", file=sys.stderr)

    counts = {"OK": 0, "ERR": 0, "NO_ABSTRACT": 0, "FAILED": 0}
    try:
        for i, row in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] id={row['id']} - {row['title'][:40]}...", file=sys.stderr)

            if not row["abstract"]:
                result = no_abstract_result(row)
            else:
                try:
                    predictions = classify_row_multi_two_stage(
                        row,
                        api_key,
                        args.reasoning_effort,
                        args.runs_per_paper,
                        args.max_retries,
                        args.retry_wait_minutes * 60,
                        args.sleep,
                    )
                    aggregate = aggregate_predictions(predictions)
                    result = format_result(row, predictions, aggregate)
                except Exception as exc:
                    print(f"  경고: 처리 실패 - {exc}", file=sys.stderr)
                    result = failed_result(row, exc)

            counts[result["status"]] = counts.get(result["status"], 0) + 1
            writer.writerow(result)
            out_f.flush()

            if i < len(todo):
                time.sleep(args.sleep)
    finally:
        out_f.close()

    print(f"\n결과 저장: {output_path}")
    print(f"이번 실행 처리: {len(todo)}편 - {counts}")


if __name__ == "__main__":
    main()
