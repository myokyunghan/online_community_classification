"""
경영학 OK 62편 재실행(논문당 3회, 과반수 채택) 결과를 기존 1회 실행 결과(classification_results.csv)와
비교한다. 논문마다 다음 셋 중 하나로 분류한다.

- 변경: 3회 과반수 판정이 기존 1회 판정과 다름 -> 기존 판정이 우연히 잘못 적용됐을 가능성
- 과반없음: 3회 판정이 전부 갈려 2회 이상 나온 라벨이 없음 -> 다수결로 못 정함, 사람 판정 대상
- 불일치: 3회 안에서 판정이 갈렸지만(full_agreement=False) 과반 라벨은 있음 -> 최종 라벨은 다수결대로
  채택하되, 경계에 있는 논문이라는 표시
- 안정: 3회 모두 같은 판정이고 기존과도 같음

비교는 ROLE 코드 집합 + 장/창 그룹 두 수준으로 한다. 장/창 그룹 기준으로 보면 SOCIALCAPITAL과
PUBLICSPHERE 사이 흔들림처럼 5b 그래프에 영향 없는 변화는 걸러진다.

사용법:
    python3 target_new/rerun_business/compare_rerun.py
    (같은 폴더에 compare_rerun.csv, 과반없음 논문이 있으면 리뷰대상_과반없음.xlsx 생성)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

from visualize_results import ROLE_FIELD_CODES, ROLE_WINDOW_CODES  # noqa: E402

OLD_CSV = HERE.parent / "classification_results.csv"
NEW_CSV = HERE / "경영학_OK_62_runs3.csv"
OUT_CSV = HERE / "compare_rerun.csv"
REVIEW_XLSX = HERE / "리뷰대상_과반없음.xlsx"
XLSX_PATH = HERE / "경영학_OK_62.xlsx"


def _codes(value) -> frozenset[str]:
    if pd.isna(value):
        return frozenset()
    return frozenset(c.strip() for c in str(value).split(",") if c.strip())


def _group(codes: frozenset[str]) -> str:
    """ROLE 코드 집합을 장/창/장+창/없음으로 요약한다."""
    has_field = bool(codes & ROLE_FIELD_CODES)
    has_window = bool(codes & ROLE_WINDOW_CODES)
    if has_field and has_window:
        return "장+창"
    return "장" if has_field else ("창" if has_window else "없음")


def _write_review_sheet(out: pd.DataFrame, new: pd.DataFrame) -> None:
    """과반없음(3회 판정이 전부 갈려 라벨이 빈칸으로 남은) 논문만 모아 사람 리뷰용 엑셀을 만든다.
    초록과 회차별 판정·근거를 나란히 놓고, 리뷰어가 채울 최종판정/메모 칸은 비워둔다."""
    review_ids = set(out.loc[out["판정"] == "과반없음", "id"])
    if not review_ids:
        print("\n과반없음 논문 없음 - 리뷰 시트 생성 안 함")
        return

    xlsx = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", dtype={"논문ID": str})
    abstracts = {
        r["논문ID"]: (r.get("KOR_ABST") if pd.notna(r.get("KOR_ABST")) else r.get("ENG_ABST"))
        for _, r in xlsx.iterrows()
    }

    rows = []
    for _, r in new[new["id"].isin(review_ids)].iterrows():
        run_codes = [part.split(":", 1)[1] for part in str(r["run_details"]).split("; ")]
        run_rationales = str(r["rationale"]).split(" | ")
        row = {"id": r["id"], "title": r["title"], "published_year": r["published_year"], "abstract": abstracts.get(r["id"])}
        for i, (codes, rationale) in enumerate(zip(run_codes, run_rationales), 1):
            row[f"run{i}_판정"] = "ERR/없음" if codes == "없음" else codes
            row[f"run{i}_근거"] = rationale
        row["최종판정(리뷰어)"] = ""
        row["메모"] = ""
        rows.append(row)

    pd.DataFrame(rows).to_excel(REVIEW_XLSX, index=False)
    print(f"\n리뷰 대상(과반없음) {len(rows)}편 -> {REVIEW_XLSX}")


def main() -> None:
    old = pd.read_csv(OLD_CSV, encoding="utf-8-sig", dtype=str)
    new = pd.read_csv(NEW_CSV, encoding="utf-8-sig", dtype=str)
    merged = new.merge(
        old[["id", "status", "role_codes", "env_codes"]],
        on="id",
        how="left",
        suffixes=("_new", "_old"),
    )

    rows = []
    for _, r in merged.iterrows():
        old_role, new_role = _codes(r["role_codes_old"]), _codes(r["role_codes_new"])
        old_group, new_group = _group(old_role), _group(new_role)
        agreed = str(r["full_agreement"]) == "True"

        if r["status_new"] == "OK" and not new_role and not _codes(r["env_codes_new"]):
            # 3회 판정이 전부 갈려서 2회 이상 나온 라벨이 하나도 없음 - 다수결로 못 정하므로 사람 판정 대상
            verdict = "과반없음"
        elif not agreed:
            verdict = "불일치"
        elif old_role != new_role or r["status_old"] != r["status_new"]:
            verdict = "변경"
        else:
            verdict = "안정"

        rows.append(
            {
                "id": r["id"],
                "title": r["title"],
                "판정": verdict,
                "장창_변경": old_group != new_group,
                "old_status": r["status_old"],
                "new_status": r["status_new"],
                "old_group": old_group,
                "new_group": new_group,
                "old_role": ", ".join(sorted(old_role)),
                "new_role": ", ".join(sorted(new_role)),
                "run_details": r["run_details"],
                "new_rationale": r["rationale"],
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    _write_review_sheet(out, new)

    print(f"비교 대상: {len(out)}편")
    print(out["판정"].value_counts().to_string())
    print(f"\n장/창 그룹이 바뀐 논문: {out['장창_변경'].sum()}편")
    print("\n[장/창 그룹 전후 교차표] (행=기존, 열=재실행)")
    print(pd.crosstab(out["old_group"], out["new_group"]).to_string())
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
