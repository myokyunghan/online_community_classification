"""
ICR/classify_paper_fewshot.py(시스템프롬프트+few-shot, 요약 단계 없음)로 golden 30편을
분류하고 정답과 비교한다. 재시도/다수결 집계/정답 비교는 루트 compare_with_golden.py의
범용 로직을 재사용한다.

few-shot 예시는 golden CSV에서 무작위로 뽑되, ERR(전부 미부여) 사례를 최소 1개는 항상
포함시킨다 — 순수 무작위로는 ERR 사례가 하나도 안 걸려서 모델이 "아무 코드도 없음" 출력
형식을 데모로 한 번도 못 보고 즉석에서 만들다 응답이 깨지는 현상이 이전 실험에서 확인됐다.
few-shot으로 뽑힌 논문은 테스트 대상에서 자동 제외된다.

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."

사용 예:
    python3 ICR/compare_with_golden_fewshot.py --sample 20 --seed 7
    python3 ICR/compare_with_golden_fewshot.py --fewshot-n 3 --seed 7 --runs-per-paper 3
    python3 ICR/compare_with_golden_fewshot.py --fewshot-n 0   # few-shot 없이(zero-shot) 비교용
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
from classify_paper_fewshot import (  # noqa: E402  (이 폴더의 버전)
    build_abstract_only_text,
    build_ideal_env_output,
    build_ideal_role_output,
    classify_paper_fewshot,
)
from compare_with_golden import (  # noqa: E402  (루트의 범용 로직 재사용)
    aggregate_predictions,
    classify_retry_kind,
    compare_row,
    load_rows,
)

ICR_DIR = Path(__file__).parent
DEFAULT_CSV = ICR_DIR / "온라인_커뮤니티_연구지형_분류논문_30편.csv"
DEFAULT_OUTPUT = ICR_DIR / "comparison_results_fewshot.csv"


def classify_row(
    row: dict,
    api_key: str,
    reasoning_effort: str,
    temperature: float = 0.0,
    fewshot_examples: list[tuple[str, dict]] | None = None,
    env_fewshot_examples: list[tuple[str, dict]] | None = None,
) -> dict:
    """classify_paper_fewshot()으로 판정하고, aggregate_predictions()/compare_row()가
    기대하는 형태로 맞춘다."""
    parsed = classify_paper_fewshot(
        build_abstract_only_text(row),
        MODEL,
        api_key,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        fewshot_examples=fewshot_examples,
        env_fewshot_examples=env_fewshot_examples,
    )
    role_codes = [item["code"] for item in parsed.get("role", [])]
    env_codes = [item["code"] for item in parsed.get("env", [])]

    division = parsed.get("division") or {}
    division_summary = (
        f"has_community={division.get('has_community')} role_relevant={division.get('role_relevant')} "
        f"env_relevant={division.get('env_relevant')} | {division.get('reasoning', '')}"
        if division
        else "(1차 판단 정보 없음)"
    )

    return {
        "parsed": parsed,
        "status": parsed.get("status", "OK"),
        "focus": division_summary,  # 1차(분리) 판단 내용을 여기 채워서 결과에서 바로 보이게 한다
        "role_codes": role_codes,
        "env_codes": env_codes,
        "predicted_codes": set(role_codes) | set(env_codes),
        "dropped_by_cap": set(parsed.get("_dropped_by_cap") or []),
    }


def classify_row_with_retry(
    row: dict,
    api_key: str,
    reasoning_effort: str,
    max_retries: int,
    retry_wait_seconds: float,
    fewshot_examples: list[tuple[str, dict]] | None = None,
    env_fewshot_examples: list[tuple[str, dict]] | None = None,
) -> dict:
    """서버 오류는 긴 대기 후, JSON 파싱 실패는 짧은 대기 후 재시도한다."""
    for attempt in range(1, max_retries + 1):
        try:
            return classify_row(
                row,
                api_key,
                reasoning_effort,
                fewshot_examples=fewshot_examples,
                env_fewshot_examples=env_fewshot_examples,
            )
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
    fewshot_examples: list[tuple[str, dict]] | None = None,
    env_fewshot_examples: list[tuple[str, dict]] | None = None,
) -> list[dict]:
    predictions = []
    for run in range(1, runs + 1):
        predictions.append(
            classify_row_with_retry(
                row,
                api_key,
                reasoning_effort,
                max_retries,
                retry_wait_seconds,
                fewshot_examples,
                env_fewshot_examples,
            )
        )
        if run < runs:
            time.sleep(sleep_seconds)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="시스템프롬프트+few-shot 방식으로 golden 30편 정확도 비교")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="정답 CSV 경로")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="비교 결과를 저장할 CSV 경로")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--limit", type=int, default=None, help="테스트용으로 앞 N편만 처리")
    parser.add_argument(
        "--ids", default=None, help="쉼표로 구분한 특정 id들만 처리 (예: --ids 4,11,18). CSV 순서를 그대로 따른다"
    )
    parser.add_argument("--sample", type=int, default=None, help="전체 중 N편을 무작위로 뽑아 테스트")
    parser.add_argument("--seed", type=int, default=None, help="--sample/--fewshot-n 사용 시 재현 가능한 무작위 시드")
    parser.add_argument(
        "--fewshot-n",
        type=int,
        default=3,
        help="golden CSV에서 무작위로 N편을 뽑아 few-shot 예시로 쓰고, 그 N편은 테스트 대상에서 제외 (기본 3, 0=zero-shot). --fewshot-ids 지정 시 무시됨",
    )
    parser.add_argument(
        "--fewshot-ids",
        default=None,
        help="쉼표로 구분한 few-shot 예시 id를 직접 고정 지정 (예: --fewshot-ids 3,11,22). 지정하면 --fewshot-n/--seed 무작위 선정 대신 이 논문들을 그대로 씀",
    )
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

    fewshot_examples = []
    env_fewshot_examples = []
    if args.fewshot_ids:
        wanted_ids = [s.strip() for s in args.fewshot_ids.split(",") if s.strip()]
        id_to_row = {r["id"]: r for r in rows}
        missing = [i for i in wanted_ids if i not in id_to_row]
        if missing:
            sys.exit(f"오류: few-shot id를 golden CSV에서 못 찾음: {', '.join(missing)}")

        fewshot_pool = [id_to_row[i] for i in wanted_ids]
        fewshot_ids = set(wanted_ids)
        rows = [r for r in rows if r["id"] not in fewshot_ids]
        fewshot_examples = [(build_abstract_only_text(r), build_ideal_role_output(r)) for r in fewshot_pool]
        env_fewshot_examples = [(build_abstract_only_text(r), build_ideal_env_output(r)) for r in fewshot_pool]
        print(
            "few-shot 예시로 사용 (고정 지정, 테스트 대상에서 제외): "
            + ", ".join(f"{r['id']}({r.get('class1', '')})" for r in fewshot_pool),
            file=sys.stderr,
        )
    elif args.fewshot_n:
        rng = random.Random(args.seed)
        # 순수 무작위로 뽑으면 ERR(전부 미부여) 사례가 하나도 안 걸릴 수 있고, 그러면 모델이
        # "아무 코드도 없음" 출력 형식을 데모로 한 번도 못 보고 즉석에서 만들다 응답이 깨지는
        # 현상이 확인됐다. 그래서 ERR 사례를 최소 1개는 항상 포함시킨다(있을 때만).
        err_rows = [r for r in rows if r.get("class1", "").strip().upper() == "ERR"]
        fewshot_pool = []
        if err_rows:
            fewshot_pool.append(rng.choice(err_rows))
        remaining = args.fewshot_n - len(fewshot_pool)
        if remaining > 0:
            picked_ids = {r["id"] for r in fewshot_pool}
            candidates = [r for r in rows if r["id"] not in picked_ids]
            fewshot_pool.extend(rng.sample(candidates, min(remaining, len(candidates))))

        fewshot_ids = {r["id"] for r in fewshot_pool}
        rows = [r for r in rows if r["id"] not in fewshot_ids]
        fewshot_examples = [(build_abstract_only_text(r), build_ideal_role_output(r)) for r in fewshot_pool]
        env_fewshot_examples = [(build_abstract_only_text(r), build_ideal_env_output(r)) for r in fewshot_pool]
        print(
            "few-shot 예시로 사용 (테스트 대상에서 제외): "
            + ", ".join(f"{r['id']}({r.get('class1', '')})" for r in fewshot_pool),
            file=sys.stderr,
        )

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
                fewshot_examples,
                env_fewshot_examples,
            )
            # 다수결(과반수, 3회면 2/3)로 최종 코드를 채택한다 — aggregate_predictions()가
            # 이미 이 계산을 해준다. full_agreement 여부는 결과 CSV에 그대로 남아있어
            # 3번 다 똑같이 나온 논문과 다수결로만 채택된 논문을 구분해볼 수 있다.
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

    # 정답과 어긋난 것(완전일치가 아닌 것)만 따로 골라 별도 CSV로 저장 + 터미널에도 바로 보여준다.
    mismatches = [r for r in evaluated if not r["exact_match"]]
    mismatch_path = output_path.parent / f"{output_path.stem}_mismatches{output_path.suffix}"
    mismatch_fieldnames = [
        "id",
        "title",
        "gold_codes",
        "majority_codes",
        "gold_covered",
        "run_details",
        "rationales",
    ]
    with mismatch_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mismatch_fieldnames)
        writer.writeheader()
        for r in mismatches:
            writer.writerow({k: r.get(k, "") for k in mismatch_fieldnames})

    if mismatches:
        print(f"\n정답과 어긋난 {len(mismatches)}편 (별도 저장: {mismatch_path}):")
        for r in mismatches:
            gold = r["gold_codes"] or "ERR"
            pred = r["majority_codes"] or "(없음/ERR)"
            tag = "부분일치(정답 포함)" if r["gold_covered"] else "오답"
            print(f"  [{tag}] id={r['id']} - {r['title'][:30]}")
            print(f"      gold={gold}")
            print(f"      pred={pred}")
    else:
        print(f"\n정답과 어긋난 논문 없음 (전부 완전일치). 빈 mismatches 파일 저장: {mismatch_path}")


if __name__ == "__main__":
    main()
