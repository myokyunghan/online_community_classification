"""
target_new/ 폴더의 실제 분류 대상 논문(KCI 원문 엑셀, KCI 사회과학.xlsx)을 golden_new의
3단계(division→role→env) 파이프라인으로 분류한다. 루트의 classify_targets.py/
classify_targets_two_stage.py(옛 11코드, NVIDIA API 단일/2단계 호출)를 대체하는 버전 —
정답(class1/class2)이 없는 실제 대상 논문이라 정답 비교는 하지 않고, 예측 role/env 코드와
근거만 CSV로 저장한다.

golden_new/classify_paper_fewshot.py의 classify_paper_fewshot()/classify_row_multi()를
그대로 재사용한다 (few-shot 예시는 golden CSV에서 고정 id로 뽑음, 기본 3,11,22).
NVIDIA_BASE_URL/MODEL/thinking on-off 등은 golden_new/classify_paper.py에서 정의된
현재 설정(2026-09-19 기준 Ollama qwen3.8:27b-mlx, thinking ON)을 그대로 따른다.

엑셀 컬럼 중 논문ID/논문명/저자/발행년/학술지 명/KOR_ABST(또는 ENG_ABST)를 사용한다.
초록이 한글/영문 둘 다 없는 논문은 API 호출 없이 status=NO_ABSTRACT로 건너뛴다.

이미 처리된 논문ID는 --output 파일에 남아있으면 기본적으로 다시 호출하지 않는다
(693편 전체를 한 번에 처리하기 어려울 수 있으므로, 중간에 끊겨도 이어서 실행 가능).
새로 처리한 논문은 한 편씩 즉시 파일에 append하므로 도중에 중단돼도 그때까지의
결과는 남는다.

사용 예:
    python3 golden_new/classify_targets.py --limit 5          # 앞 5편만 테스트
    python3 golden_new/classify_targets.py                    # 전체 실행 (이어하기 지원)
    python3 golden_new/classify_targets.py --overwrite        # 처음부터 다시
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classify_paper_fewshot import (  # noqa: E402
    build_abstract_only_text,
    build_ideal_division_output,
    build_ideal_env_output,
    build_ideal_role_output,
)
from compare_with_golden import aggregate_predictions, load_rows  # noqa: E402  (루트, 범용 로직 재사용)
from compare_with_golden_fewshot import DUMMY_API_KEY, classify_row_multi  # noqa: E402  (이 폴더 버전)

GOLDEN_NEW_DIR = Path(__file__).parent
DEFAULT_GOLDEN_CSV = GOLDEN_NEW_DIR / "온라인_커뮤니티_연구지형_분류논문_30편.csv"
TARGET_DIR = GOLDEN_NEW_DIR.parent / "target_new"
DEFAULT_XLSX = TARGET_DIR / "KCI 사회과학.xlsx"
DEFAULT_OUTPUT = TARGET_DIR / "classification_results.csv"

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


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_target_rows(xlsx_path: Path, sheet: str = "Sheet1") -> list[dict]:
    """엑셀을 읽어 build_abstract_only_text()가 기대하는 키(abstract 포함)로 변환한다.
    KOR_ABST가 없으면 ENG_ABST로 대체하고, 둘 다 없으면 abstract를 빈 문자열로 둔다
    (호출 쪽에서 abstract가 빈 문자열이면 API 호출 없이 NO_ABSTRACT로 건너뛴다)."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    rows = []
    for _, r in df.iterrows():
        kor_abst = _clean(r.get("KOR_ABST"))
        eng_abst = _clean(r.get("ENG_ABST"))
        if kor_abst:
            abstract, lang = kor_abst, "kor"
        elif eng_abst:
            abstract, lang = eng_abst, "eng"
        else:
            abstract, lang = "", "none"

        title = _clean(r.get("논문명")) or _clean(r.get("논문 외국어명"))

        rows.append(
            {
                "id": _clean(r.get("논문ID")),
                "title": title,
                "author": _clean(r.get("저자")),
                "published_year": _clean(r.get("발행년")),
                "journal": _clean(r.get("학술지 명")),
                "abstract": abstract,
                "abstract_lang": lang,
            }
        )
    return rows


def load_processed_ids(output_path: Path) -> set[str]:
    """FAILED(서버 타임아웃 등 일시적 오류로 끝내 실패)는 "처리됨"으로 치지 않는다 - 다음 실행
    때 자동으로 재시도되게 한다. NO_ABSTRACT/OK/ERR만 진짜로 끝난 것으로 취급한다."""
    if not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8-sig") as f:
        return {row["id"] for row in csv.DictReader(f) if row.get("status") != "FAILED"}


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


def no_abstract_result(row: dict) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "author": row["author"],
        "published_year": row["published_year"],
        "journal": row["journal"],
        "abstract_lang": "none",
        "status": "NO_ABSTRACT",
        "role_codes": "",
        "env_codes": "",
        "dropped_by_cap": "",
        "focus": "",
        "rationale": "초록 없음 (KOR_ABST/ENG_ABST 둘 다 없음)",
        "full_agreement": "",
        "error_runs": "",
        "run_details": "",
    }


def failed_result(row: dict, exc: Exception) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "author": row["author"],
        "published_year": row["published_year"],
        "journal": row["journal"],
        "abstract_lang": row["abstract_lang"],
        "status": "FAILED",
        "role_codes": "",
        "env_codes": "",
        "dropped_by_cap": "",
        "focus": "",
        "rationale": f"처리 실패: {exc}",
        "full_agreement": "",
        "error_runs": "",
        "run_details": "",
    }


def main():
    parser = argparse.ArgumentParser(description="target_new/ 폴더 실제 논문을 golden_new 3단계 파이프라인으로 분류")
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="분류 대상 엑셀 경로")
    parser.add_argument("--sheet", default="Sheet1", help="엑셀 시트 이름")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="결과를 저장할 CSV 경로")
    parser.add_argument("--golden-csv", default=str(DEFAULT_GOLDEN_CSV), help="few-shot 예시를 뽑아올 golden CSV 경로")
    parser.add_argument(
        "--fewshot-ids", default="3,11,22", help="쉼표로 구분한 few-shot 예시 id (고정, 기본 3,11,22)"
    )
    parser.add_argument("--api-key", default=None, help="자체 서버 API 키 (인증 불필요 - 미지정 시 더미 값 사용)")
    parser.add_argument("--limit", type=int, default=None, help="테스트용으로 앞 N편만 처리")
    parser.add_argument("--start", type=int, default=0, help="이 순번부터 처리 (0-based, 기본 0)")
    parser.add_argument("--runs-per-paper", type=int, default=1, help="논문당 반복 호출 횟수, 과반수로 채택 (기본 1)")
    parser.add_argument(
        "--reasoning-effort", default="low", choices=["low", "medium", "high"], help="모델의 reasoning_effort (기본 low, 현재 서버엔 미적용)"
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="API 호출 사이 대기 시간(초, 기본 1.0)")
    parser.add_argument("--max-retries", type=int, default=3, help="일시적 오류/파싱 실패 시 최대 재시도 횟수 (기본 3)")
    parser.add_argument("--retry-wait-minutes", type=float, default=3.0, help="서버 오류 재시도 전 대기 시간(분, 기본 3.0)")
    parser.add_argument(
        "--overwrite", action="store_true", help="기존 --output 파일을 무시하고 처음부터 다시 처리"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY") or DUMMY_API_KEY

    golden_rows = load_rows(Path(args.golden_csv))
    wanted_ids = [s.strip() for s in args.fewshot_ids.split(",") if s.strip()]
    id_to_row = {r["id"]: r for r in golden_rows}
    missing = [i for i in wanted_ids if i not in id_to_row]
    if missing:
        sys.exit(f"오류: few-shot id를 golden CSV에서 못 찾음: {', '.join(missing)}")
    fewshot_pool = [id_to_row[i] for i in wanted_ids]
    fewshot_examples = [(build_abstract_only_text(r), build_ideal_role_output(r)) for r in fewshot_pool]
    env_fewshot_examples = [(build_abstract_only_text(r), build_ideal_env_output(r)) for r in fewshot_pool]
    division_fewshot_examples = [(build_abstract_only_text(r), build_ideal_division_output(r)) for r in fewshot_pool]
    print(
        "few-shot 고정: " + ", ".join(f"{r['id']}({r.get('class1', '')})" for r in fewshot_pool),
        file=sys.stderr,
    )

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
                        division_fewshot_examples,
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
