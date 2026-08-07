"""
같은 평가(golden CSV에서 few-shot 예시로 뽑은 논문 제외 나머지 전체, runs-per-paper 다수결)를
N번 반복해서, 반복 간 평균 정확도·F1을 계산한다. 매 반복은 --runs-per-paper(기본 3)로
다수결을 적용한 뒤, 그 다수결 결과 자체가 반복마다 얼마나 안정적인지를 다시 N번 모아
평균·표준편차로 본다.

각 반복의 논문별 원자료는 --raw-output(기본 repeat_eval_raw.csv)에 반복마다 즉시 append하고
flush한다 — 중간에 중단돼도 그때까지 완료된 반복은 그대로 남는다. --start-repeat으로
중단된 지점부터 이어서 실행할 수 있다.

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."

사용 예:
    python3 ICR/repeat_eval.py --repeats 100 --fewshot-ids 3,11,22 --runs-per-paper 3
    python3 ICR/repeat_eval.py --repeats 100 --start-repeat 37   # 36번까지 끝난 뒤 이어서
"""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classify_paper_fewshot import build_abstract_only_text, build_ideal_env_output, build_ideal_role_output  # noqa: E402
from compare_with_golden import aggregate_predictions, compare_row, load_rows  # noqa: E402
from compare_with_golden_fewshot import classify_row_multi  # noqa: E402  (이 폴더의 버전 재사용)

ICR_DIR = Path(__file__).parent
DEFAULT_CSV = ICR_DIR / "온라인_커뮤니티_연구지형_분류논문_30편.csv"
DEFAULT_RAW_OUTPUT = ICR_DIR / "repeat_eval_raw.csv"
DEFAULT_SUMMARY_OUTPUT = ICR_DIR / "repeat_eval_summary.csv"

RAW_FIELDNAMES = [
    "repeat",
    "id",
    "title",
    "is_err_gold",
    "gold_codes",
    "majority_codes",
    "majority_status",
    "exact_match",
    "gold_covered",
    "precision",
    "recall",
    "f1",
    "full_agreement",
]


def _code_set(s: str) -> set:
    return {c.strip() for c in (s or "").split(",") if c.strip()}


def _compute_prf1(gold: set, pred: set) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    inter = gold & pred
    precision = len(inter) / len(pred) if pred else 0.0
    recall = len(inter) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def run_one_repeat(
    rep: int,
    test_rows: list[dict],
    api_key: str,
    reasoning_effort: str,
    runs_per_paper: int,
    max_retries: int,
    retry_wait_seconds: float,
    sleep_seconds: float,
    fewshot_examples: list,
    env_fewshot_examples: list,
    raw_writer: csv.DictWriter,
    raw_file,
) -> list[dict]:
    rep_rows = []
    for i, row in enumerate(test_rows, 1):
        print(f"  [{i}/{len(test_rows)}] id={row['id']} - {row['title'][:30]}...", file=sys.stderr)
        is_err_gold = row.get("class1", "").strip().upper() == "ERR"
        try:
            predictions = classify_row_multi(
                row,
                api_key,
                reasoning_effort,
                runs_per_paper,
                max_retries,
                retry_wait_seconds,
                sleep_seconds,
                fewshot_examples,
                env_fewshot_examples,
            )
            aggregate = aggregate_predictions(predictions)
            result = compare_row(row, predictions, aggregate)
        except Exception as exc:  # API 오류, JSON 파싱 실패 등 — 이 논문은 이번 반복에서 실패로 기록
            print(f"    경고: 처리 실패 - {exc}", file=sys.stderr)
            result = {
                "gold_codes": "",
                "majority_status": "",
                "majority_codes": "",
                "exact_match": False,
                "gold_covered": False,
                "full_agreement": "",
            }

        gold = _code_set(result.get("gold_codes", ""))
        pred = _code_set(result.get("majority_codes", ""))
        if is_err_gold:
            # ERR 케이스는 코드 집합 비교가 의미 없으므로, 상태 일치 여부를 1.0/0.0으로 환산해
            # precision/recall/f1에 그대로 반영한다 (맞았으면 1.0, 아니면 0.0).
            correct = result.get("exact_match") is True
            precision = recall = f1 = 1.0 if correct else 0.0
        else:
            precision, recall, f1 = _compute_prf1(gold, pred)

        row_out = {
            "repeat": rep,
            "id": row["id"],
            "title": row["title"],
            "is_err_gold": is_err_gold,
            "gold_codes": result.get("gold_codes", ""),
            "majority_codes": result.get("majority_codes", ""),
            "majority_status": result.get("majority_status", ""),
            "exact_match": result.get("exact_match"),
            "gold_covered": result.get("gold_covered"),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "full_agreement": result.get("full_agreement", ""),
        }
        raw_writer.writerow(row_out)
        raw_file.flush()
        rep_rows.append(row_out)

        if i < len(test_rows):
            time.sleep(sleep_seconds)

    return rep_rows


def summarize_repeat(rep: int, rep_rows: list[dict]) -> dict:
    normal = [r for r in rep_rows if not r["is_err_gold"]]
    err_rows = [r for r in rep_rows if r["is_err_gold"]]

    exact = sum(1 for r in normal if r["exact_match"] is True)
    covered = sum(1 for r in normal if r["gold_covered"] is True)
    err_correct = sum(1 for r in err_rows if r["exact_match"] is True)
    combined_correct = exact + err_correct
    combined_covered = covered + err_correct

    avg_f1 = statistics.mean(r["f1"] for r in rep_rows) if rep_rows else 0.0
    avg_precision = statistics.mean(r["precision"] for r in rep_rows) if rep_rows else 0.0
    avg_recall = statistics.mean(r["recall"] for r in rep_rows) if rep_rows else 0.0

    return {
        "repeat": rep,
        "n_total": len(rep_rows),
        "n_normal": len(normal),
        "n_err": len(err_rows),
        "exact_normal": exact,
        "covered_normal": covered,
        "err_correct": err_correct,
        "combined_exact_rate": round(combined_correct / len(rep_rows), 4) if rep_rows else 0.0,
        "combined_covered_rate": round(combined_covered / len(rep_rows), 4) if rep_rows else 0.0,
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "avg_f1": round(avg_f1, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="golden 평가를 N번 반복해서 평균 정확도/F1을 계산")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="정답 CSV 경로")
    parser.add_argument("--repeats", type=int, default=100, help="반복 횟수 (기본 100)")
    parser.add_argument("--start-repeat", type=int, default=1, help="이 번호부터 시작 (중단 후 재시작용, 기본 1)")
    parser.add_argument(
        "--fewshot-ids", default="3,11,22", help="쉼표로 구분한 few-shot 예시 id (고정, 기본 3,11,22)"
    )
    parser.add_argument("--runs-per-paper", type=int, default=3, help="논문당 반복 호출 횟수, 과반수로 채택 (기본 3)")
    parser.add_argument(
        "--reasoning-effort", default="low", choices=["low", "medium", "high"], help="모델의 reasoning_effort (기본 low)"
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="API 호출 사이 대기 시간(초, 기본 1.0)")
    parser.add_argument("--max-retries", type=int, default=3, help="타임아웃/연결 오류 시 최대 재시도 횟수 (기본 3)")
    parser.add_argument("--retry-wait-minutes", type=float, default=3.0, help="재시도 전 대기 시간(분, 기본 3.0)")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT), help="반복×논문 단위 원자료 CSV 경로")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT), help="반복별 요약 CSV 경로")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

    rows = load_rows(Path(args.csv))
    wanted_ids = [s.strip() for s in args.fewshot_ids.split(",") if s.strip()]
    id_to_row = {r["id"]: r for r in rows}
    missing = [i for i in wanted_ids if i not in id_to_row]
    if missing:
        sys.exit(f"오류: few-shot id를 golden CSV에서 못 찾음: {', '.join(missing)}")

    fewshot_pool = [id_to_row[i] for i in wanted_ids]
    fewshot_id_set = set(wanted_ids)
    test_rows = [r for r in rows if r["id"] not in fewshot_id_set]
    fewshot_examples = [(build_abstract_only_text(r), build_ideal_role_output(r)) for r in fewshot_pool]
    env_fewshot_examples = [(build_abstract_only_text(r), build_ideal_env_output(r)) for r in fewshot_pool]

    print(
        f"few-shot 고정: {', '.join(wanted_ids)} / 테스트 대상: {len(test_rows)}편 / "
        f"반복: {args.start_repeat}~{args.repeats} / runs-per-paper: {args.runs_per_paper}",
        file=sys.stderr,
    )

    raw_path = Path(args.raw_output)
    write_header = args.start_repeat == 1 or not raw_path.exists()
    raw_file = raw_path.open("a" if not write_header else "w", encoding="utf-8-sig", newline="")
    raw_writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDNAMES)
    if write_header:
        raw_writer.writeheader()

    summary_path = Path(args.summary_output)
    summary_write_header = args.start_repeat == 1 or not summary_path.exists()
    summary_file = summary_path.open("a" if not summary_write_header else "w", encoding="utf-8-sig", newline="")
    summary_fieldnames = [
        "repeat",
        "n_total",
        "n_normal",
        "n_err",
        "exact_normal",
        "covered_normal",
        "err_correct",
        "combined_exact_rate",
        "combined_covered_rate",
        "avg_precision",
        "avg_recall",
        "avg_f1",
    ]
    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fieldnames)
    if summary_write_header:
        summary_writer.writeheader()

    all_summaries = []
    try:
        for rep in range(args.start_repeat, args.repeats + 1):
            print(f"\n=== 반복 {rep}/{args.repeats} ===", file=sys.stderr)
            rep_rows = run_one_repeat(
                rep,
                test_rows,
                api_key,
                args.reasoning_effort,
                args.runs_per_paper,
                args.max_retries,
                args.retry_wait_minutes * 60,
                args.sleep,
                fewshot_examples,
                env_fewshot_examples,
                raw_writer,
                raw_file,
            )
            summary = summarize_repeat(rep, rep_rows)
            summary_writer.writerow(summary)
            summary_file.flush()
            all_summaries.append(summary)
            print(
                f"  반복 {rep} 결과: 합산정답률={summary['combined_exact_rate']:.2%} "
                f"정답포함율={summary['combined_covered_rate']:.2%} 평균F1={summary['avg_f1']:.3f}",
                file=sys.stderr,
            )
    finally:
        raw_file.close()
        summary_file.close()

    if not all_summaries:
        print("완료된 반복이 없습니다.", file=sys.stderr)
        return

    exact_rates = [s["combined_exact_rate"] for s in all_summaries]
    covered_rates = [s["combined_covered_rate"] for s in all_summaries]
    f1s = [s["avg_f1"] for s in all_summaries]

    def _mean_std(values):
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        return mean, std

    exact_mean, exact_std = _mean_std(exact_rates)
    covered_mean, covered_std = _mean_std(covered_rates)
    f1_mean, f1_std = _mean_std(f1s)

    print(f"\n=== 이번 실행에서 완료된 {len(all_summaries)}회 반복 종합 (반복 {args.start_repeat}~{args.start_repeat + len(all_summaries) - 1}) ===")
    print(f"합산 정답률(완전일치+ERR정답): 평균 {exact_mean:.2%}, 표준편차 {exact_std:.2%}")
    print(f"합산 포함율(정답포함+ERR정답): 평균 {covered_mean:.2%}, 표준편차 {covered_std:.2%}")
    print(f"평균 F1: 평균 {f1_mean:.3f}, 표준편차 {f1_std:.3f}")
    print(f"\n원자료(반복×논문): {raw_path}")
    print(f"반복별 요약: {summary_path}")


if __name__ == "__main__":
    main()
