"""
target/ 폴더의 실제 분류 대상 논문(KCI 원문 엑셀)을 NVIDIA NIM API로 분류한다.
compare_with_golden.py와 달리 정답(class1/class2)이 없으므로 정답 비교는 하지 않고,
각 논문의 예측 role/env 코드와 근거만 CSV로 저장한다.

엑셀 컬럼 중 논문ID/논문명/저자/발행년/학술지 명/KOR_ABST(또는 ENG_ABST)를 사용해
compare_with_golden.py의 build_paper_text()가 기대하는 형태의 row로 변환한 뒤,
같은 classify_row_multi()/aggregate_predictions()를 그대로 재사용한다 (로직 중복 방지).

초록이 한글/영문 둘 다 없는 논문은 API 호출 없이 status=NO_ABSTRACT로 건너뛴다.

이미 처리된 논문ID는 --output 파일에 남아있으면 기본적으로 다시 호출하지 않는다
(693편 전체를 한 번에 처리하기 어려울 수 있으므로, 중간에 끊겨도 이어서 실행 가능).
새로 처리한 논문은 한 편씩 즉시 파일에 append하므로 도중에 중단돼도 그때까지의
결과는 남는다.

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."
    pip install -r requirements.txt   # pandas, openpyxl 포함

사용 예:
    python classify_targets.py
    python classify_targets.py --limit 5                 # 앞 5편만 테스트
    python classify_targets.py --runs-per-paper 3         # 논문당 3회 호출 후 다수결
    python classify_targets.py --overwrite                # 기존 결과 무시하고 처음부터
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

from compare_with_golden import aggregate_predictions, classify_row_multi

DEFAULT_XLSX = Path(__file__).parent / "target" / "KCI 사회과학.xlsx"
DEFAULT_OUTPUT = Path(__file__).parent / "target" / "classification_results.csv"

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
    """엑셀을 읽어 build_paper_text()가 기대하는 키(title/author/published_year/journal/abstract)로
    변환한다. KOR_ABST가 없으면 ENG_ABST로 대체하고, 둘 다 없으면 abstract를 빈 문자열로 둔다
    (호출 쪽에서 abstract_lang == "none"으로 걸러 API 호출을 건너뛴다)."""
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
    if not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8-sig") as f:
        return {row["id"] for row in csv.DictReader(f)}


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
    parser = argparse.ArgumentParser(description="target/ 폴더 실제 논문 NVIDIA API 분류")
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="분류 대상 엑셀 경로")
    parser.add_argument("--sheet", default="Sheet1", help="엑셀 시트 이름")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="결과를 저장할 CSV 경로")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--limit", type=int, default=None, help="테스트용으로 앞 N편만 처리")
    parser.add_argument("--start", type=int, default=0, help="이 순번부터 처리 (0-based, 기본 0)")
    parser.add_argument("--runs-per-paper", type=int, default=1, help="논문당 반복 호출 횟수, 과반수로 채택 (기본 1)")
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
                    predictions = classify_row_multi(
                        row,
                        api_key,
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
