"""
ICR/ 폴더의 판정 기준을 단일 호출(prompt_template_single_stage.txt)로 golden 30편과 비교한다.
compare_with_golden_two_stage.py와 재시도/다수결 집계/정답 비교 로직은 동일하게 루트의
compare_with_golden.py 것을 재사용하고, classify_row()만 classify_paper_single_stage()를
쓰도록 바꿔 2단계 방식과 나란히 비교할 수 있게 했다.

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."

사용 예:
    python3 ICR/compare_with_golden_single_stage.py --limit 3
    python3 ICR/compare_with_golden_single_stage.py --sample 20
    python3 ICR/compare_with_golden_single_stage.py --ids 3,18,22
    python3 ICR/compare_with_golden_single_stage.py --runs-per-paper 3
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classify_paper import MODEL  # noqa: E402
from classify_paper_single_stage import classify_paper_single_stage  # noqa: E402  (이 폴더의 버전)
from compare_with_golden import (  # noqa: E402  (루트의 범용 로직 재사용)
    aggregate_predictions,
    build_paper_text,
    classify_retry_kind,
    compare_row,
    load_rows,
)

ICR_DIR = Path(__file__).parent
DEFAULT_CSV = ICR_DIR / "온라인_커뮤니티_연구지형_분류논문_30편.csv"
DEFAULT_OUTPUT = ICR_DIR / "comparison_results_single_stage.csv"


def classify_row(row: dict, api_key: str, reasoning_effort: str, temperature: float = 0.0) -> dict:
    """단일 호출용 classify_row() — classify_paper_single_stage()로 한 번에 판정하고,
    aggregate_predictions()/compare_row()가 기대하는 형태로 맞춘다."""
    parsed = classify_paper_single_stage(
        build_paper_text(row), MODEL, api_key, temperature=temperature, reasoning_effort=reasoning_effort
    )
    role_codes = [item["code"] for item in parsed.get("role", [])]
    env_codes = [item["code"] for item in parsed.get("env", [])]

    return {
        "parsed": parsed,
        "status": parsed.get("status", "OK"),
        "focus": parsed.get("focus"),
        "role_codes": role_codes,
        "env_codes": env_codes,
        "predicted_codes": set(role_codes) | set(env_codes),
        "dropped_by_cap": set(parsed.get("_dropped_by_cap") or []),
    }


def classify_row_with_retry(
    row: dict, api_key: str, reasoning_effort: str, max_retries: int, retry_wait_seconds: float
) -> dict:
    """서버 오류는 긴 대기 후, JSON 파싱 실패는 짧은 대기 후 재시도한다."""
    for attempt in range(1, max_retries + 1):
        try:
            return classify_row(row, api_key, reasoning_effort)
        except Exception as exc:
            kind = classify_retry_kind(exc)
            if kind is None or attempt >= max_retries:
                raise
            wait = retry_wait_seconds if kind == "server" else 2.0
            label = f"{retry_wait_seconds / 60:.1f}분" if kind == "server" else "2초"
            print(f"    일시적 오류 (시도 {attempt}/{max_retries}) - {label} 후 재시도: {exc}", file=sys.stderr)
            time.sleep(wait)


def classify_row_multi(
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
            classify_row_with_retry(row, api_key, reasoning_effort, max_retries, retry_wait_seconds)
        )
        if run < runs:
            time.sleep(sleep_seconds)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="ICR 판정 기준을 단일 호출로 golden 30편 정확도 비교")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="정답 CSV 경로")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="비교 결과를 저장할 CSV 경로")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--limit", type=int, default=None, help="테스트용으로 앞 N편만 처리")
    parser.add_argument(
        "--ids", default=None, help="쉼표로 구분한 특정 id들만 처리 (예: --ids 4,11,18). CSV 순서를 그대로 따른다"
    )
    parser.add_argument("--sample", type=int, default=None, help="전체 중 N편을 무작위로 뽑아 테스트")
    parser.add_argument("--seed", type=int, default=None, help="--sample 사용 시 재현 가능한 무작위 시드")
    parser.add_argument("--runs-per-paper", type=int, default=1, help="논문당 반복 호출 횟수, 과반수로 채택 (기본 1)")
    parser.add_argument(
        "--reasoning-effort", default="low", choices=["low", "medium", "high"], help="모델의 reasoning_effort (기본 low)"
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="API 호출 사이 대기 시간(초, 기본 1.0)")
    parser.add_argument("--max-retries", type=int, default=3, help="타임아웃/연결 오류 시 최대 재시도 횟수 (기본 3)")
    parser.add_argument("--retry-wait-minutes", type=float, default=3.0, help="재시도 전 대기 시간(분, 기본 3.0)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

    rows = load_rows(Path(args.csv))
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        rows = [r for r in rows if r["id"] in wanted]
        missing = wanted - {r["id"] for r in rows}
        if missing:
            print(f"경고: golden CSV에서 못 찾은 id: {', '.join(sorted(missing))}", file=sys.stderr)
    elif args.sample:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, min(args.sample, len(rows)))
    elif args.limit:
        rows = rows[: args.limit]

    results = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] id={row['id']} - {row['title'][:30]}...", file=sys.stderr)
        try:
            predictions = classify_row_multi(
                row,
                api_key,
                args.reasoning_effort,
                args.runs_per_paper,
                args.max_retries,
                args.retry_wait_minutes * 60,
                args.sleep,
            )
            aggregate = aggregate_predictions(predictions)
            results.append(compare_row(row, predictions, aggregate))
        except Exception as exc:  # API 오류, JSON 파싱 실패 등
            print(f"  경고: 처리 실패 - {exc}", file=sys.stderr)
            results.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "class1_raw": row.get("class1", ""),
                    "class1_code": "",
                    "class2_raw": row.get("class2", ""),
                    "class2_code": "",
                    "gold_codes": "",
                    "majority_status": "",
                    "majority_codes": "",
                    "run_details": "",
                    "full_agreement": "",
                    "error_runs": "",
                    "dropped_by_cap_runs": "",
                    "gold_dropped_by_cap": "",
                    "exact_match": False,
                    "gold_covered": False,
                    "focuses": "",
                    "rationales": "",
                }
            )
        if i < len(rows):
            time.sleep(args.sleep)

    fieldnames = list(results[0].keys())
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    evaluated = [r for r in results if r["gold_codes"]]
    err_gold = [r for r in evaluated if r["class1_raw"].upper() == "ERR"]
    normal = [r for r in evaluated if r["class1_raw"].upper() != "ERR"]

    exact_correct = sum(1 for r in normal if r["exact_match"])
    covered_correct = sum(1 for r in normal if r["gold_covered"])
    agreed = sum(1 for r in evaluated if r["full_agreement"] is True)
    any_error = sum(1 for r in results if r.get("error_runs") and not r["error_runs"].startswith("0/"))
    err_correct = sum(1 for r in err_gold if r["exact_match"])
    gold_dropped = sum(1 for r in normal if r.get("gold_dropped_by_cap") is True)

    print(f"\n결과 저장: {output_path}")
    print(f"완전 일치(정답 집합 == 예측 집합, ERR 테스트 케이스 제외): {exact_correct}/{len(normal)}")
    print(f"정답 카테고리 포함(예측이 정답을 모두 포함, 추가 예측은 허용): {covered_correct}/{len(normal)}")
    print(f"{args.runs_per_paper}회 호출 전부 동일한 결과(완전 합의): {agreed}/{len(evaluated)}")
    print(f"적어도 한 번은 ERR(판단 불가) 응답이 나온 논문: {any_error}/{len(results)}")
    print(f"'합쳐서 최대 2개' 캡이 정답 코드를 잘라낸 논문: {gold_dropped}/{len(normal)}")
    if err_gold:
        print(f"ERR 정답 케이스(온라인 커뮤니티 논문이 아님)를 올바르게 ERR로 판정: {err_correct}/{len(err_gold)}")


if __name__ == "__main__":
    main()
