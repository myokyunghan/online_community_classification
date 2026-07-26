"""
golden/comparison_results.csv에서 골든 정답과 LLM 예측이 갈리는(exact_match=False)
논문만 골라 golden/mismatches.csv로 저장한다. 원본 golden CSV의 초록도 함께 붙여서
바로 검토할 수 있게 한다.

사전 준비: compare_with_golden.py를 먼저 실행해 comparison_results.csv를 만들어둔다.

사용 예:
    python find_mismatches.py
"""

import csv
from pathlib import Path

COMPARISON_CSV = Path(__file__).parent / "golden" / "comparison_results.csv"
GOLDEN_CSV = Path(__file__).parent / "golden" / "온라인_커뮤니티_연구지형_분류논문_22편.csv"
OUTPUT_CSV = Path(__file__).parent / "golden" / "mismatches.csv"


def main():
    with COMPARISON_CSV.open(encoding="utf-8-sig") as f:
        comparison_rows = list(csv.DictReader(f))

    with GOLDEN_CSV.open(encoding="utf-8-sig") as f:
        golden_by_id = {row["id"]: row for row in csv.DictReader(f)}

    mismatches = []
    for row in comparison_rows:
        if row.get("exact_match") == "True":
            continue
        golden_row = golden_by_id.get(row["id"], {})
        mismatches.append(
            {
                "id": row["id"],
                "title": row["title"],
                "author": golden_row.get("author", ""),
                "published_year": golden_row.get("published_year", ""),
                "class1_raw": row["class1_raw"],
                "class2_raw": row["class2_raw"],
                "gold_codes": row["gold_codes"],
                "majority_status": row.get("majority_status", ""),
                "majority_codes": row["majority_codes"],
                "gold_covered": row["gold_covered"],
                "full_agreement": row["full_agreement"],
                "error_runs": row.get("error_runs", ""),
                "dropped_by_cap_runs": row.get("dropped_by_cap_runs", ""),
                "gold_dropped_by_cap": row.get("gold_dropped_by_cap", ""),
                "run_details": row["run_details"],
                "focuses": row.get("focuses", ""),
                "rationales": row["rationales"],
                "abstract": golden_row.get("abstract", ""),
            }
        )

    if not mismatches:
        print("불일치 항목이 없습니다.")
        return

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mismatches[0].keys()))
        writer.writeheader()
        writer.writerows(mismatches)

    print(f"불일치 {len(mismatches)}건 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
