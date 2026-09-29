"""
1회 분류 결과(classification_results.csv)와 3회 다수결 결과(classification_results_runs3.csv)의 최종
판정을 논문별로 비교해서, 판정이 달라진 논문만 뽑아 엑셀로 정리한다.

최종 판정은 "ERR" 또는 코드 집합(ROLE+ENV)이다. 3회 결과에서 과반 라벨이 없는 논문은 "(과반없음)"으로
본다. 초록 없는 논문은 두 결과 모두 NO_ABSTRACT라 비교에서 뺀다.

달라진 논문을 다음 유형 중 하나로 나눈다(위에서부터 먼저 해당하는 것).
- OK→ERR: 1회에는 커뮤니티 연구로 코드가 붙었는데 3회 다수결은 ERR
- ERR→OK: 반대
- →과반없음: 3회가 전부 갈려 라벨이 비었음
- 그룹 변경: 둘 다 OK인데 장(ROLE_FIELD)/창(ROLE_WINDOW)/환경(ENV) 그룹 구성이 달라짐
  (예: 장 -> 창, 창 -> 창+환경)
- 그룹 내 코드 변경: 그룹 구성은 같고 세부 코드만 달라짐 (예: MISC -> GENDER, HATE -> GENDER+HATE)

사용법:
    python3 target_new/compare_run1_runs3.py
    (target_new/비교_1회_vs_3회.xlsx 생성 - 시트: 요약 / 분야별 / 그룹전이 / 변경목록)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RUN1_CSV = HERE / "classification_results.csv"
RUNS3_CSV = HERE / "classification_results_runs3.csv"
XLSX_PATH = HERE / "KCI 사회과학.xlsx"
OUT_XLSX = HERE / "비교_1회_vs_3회.xlsx"

ALLOWED_FIELDS = [
    "경제학", "기타사회과학", "사회과학", "사회과학일반", "사회복지학",
    "사회학", "신문방송학", "심리과학", "인류학", "정치외교학", "지리학", "경영학",
]
CHANGE_ORDER = ["OK→ERR", "ERR→OK", "→과반없음", "그룹 변경", "그룹 내 코드 변경"]


def _codes(row: pd.Series) -> frozenset[str]:
    return frozenset(
        c.strip()
        for col in ("role_codes", "env_codes")
        for c in str(row[col]).split(",")
        if c.strip() and c.strip() != "nan"
    )


def _final(row: pd.Series) -> str:
    if row["status"] != "OK":
        return row["status"]
    codes = _codes(row)
    return ", ".join(sorted(codes)) if codes else "(과반없음)"


def _groups(final: str) -> str:
    """최종 판정을 장/창/환경 그룹 구성으로 요약한다 (예: '창+환경'). ERR 등은 그대로."""
    if final in ("ERR", "NO_ABSTRACT", "(과반없음)"):
        return final
    parts = []
    if "ROLE_FIELD" in final:
        parts.append("장")
    if "ROLE_WINDOW" in final:
        parts.append("창")
    if "ENV_" in final:
        parts.append("환경")
    return "+".join(parts)


def _change_type(before: str, after: str) -> str:
    if before != "ERR" and after == "ERR":
        return "OK→ERR"
    if before == "ERR" and after != "ERR":
        return "→과반없음" if after == "(과반없음)" else "ERR→OK"
    if after == "(과반없음)":
        return "→과반없음"
    if _groups(before) != _groups(after):
        return "그룹 변경"
    return "그룹 내 코드 변경"


def _split_runs(text: str) -> list[str]:
    """3회분 근거를 회차별로 나눈다 - analyze_runs3.py와 같은 규칙."""
    runs: list[str] = []
    for token in str(text).split(" | "):
        if token.startswith("[ENV]") and runs and runs[-1].startswith("[ROLE]") and "[ENV]" not in runs[-1]:
            runs[-1] += " | " + token
        else:
            runs.append(token)
    return runs


def main() -> None:
    run1 = pd.read_csv(RUN1_CSV, encoding="utf-8-sig", dtype=str)
    runs3 = pd.read_csv(RUNS3_CSV, encoding="utf-8-sig", dtype=str)
    meta = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", dtype={"논문ID": str})
    meta = meta.rename(columns={"논문ID": "id", "중분류": "분야"})
    meta["초록"] = meta["KOR_ABST"].where(meta["KOR_ABST"].notna(), meta["ENG_ABST"])

    merged = run1[["id", "title", "published_year", "status", "role_codes", "env_codes", "rationale"]].merge(
        runs3[["id", "status", "role_codes", "env_codes", "rationale", "run_details"]],
        on="id",
        suffixes=("_1", "_3"),
    )
    merged["1회_판정"] = merged.rename(
        columns={"status_1": "status", "role_codes_1": "role_codes", "env_codes_1": "env_codes"}
    ).apply(_final, axis=1)
    merged["3회_판정"] = merged.rename(
        columns={"status_3": "status", "role_codes_3": "role_codes", "env_codes_3": "env_codes"}
    ).apply(_final, axis=1)
    merged = merged[merged["1회_판정"] != "NO_ABSTRACT"]
    merged = merged.merge(meta[["id", "분야", "초록"]], on="id", how="left")
    merged["화이트리스트"] = merged["분야"].isin(ALLOWED_FIELDS)
    merged["1회_그룹"] = merged["1회_판정"].apply(_groups)
    merged["3회_그룹"] = merged["3회_판정"].apply(_groups)

    changed = merged[merged["1회_판정"] != merged["3회_판정"]].copy()
    changed["변경유형"] = [_change_type(b, a) for b, a in zip(changed["1회_판정"], changed["3회_판정"])]
    changed["변경유형"] = pd.Categorical(changed["변경유형"], CHANGE_ORDER, ordered=True)

    for i in range(3):
        changed[f"run{i + 1}_판정"] = changed["run_details"].apply(
            lambda s, i=i: str(s).split("; ")[i].split(":", 1)[1].replace("없음", "ERR/없음")
        )
        changed[f"run{i + 1}_근거"] = changed["rationale_3"].apply(lambda s, i=i: _split_runs(s)[i])

    changed = changed.sort_values(["변경유형", "분야", "1회_판정"])
    columns = [
        "id", "분야", "화이트리스트", "title", "published_year", "변경유형", "1회_판정", "3회_판정",
        "1회_그룹", "3회_그룹", "run1_판정", "run2_판정", "run3_판정", "초록",
        "rationale_1", "run1_근거", "run2_근거", "run3_근거",
    ]
    detail = changed[columns].rename(
        columns={"title": "제목", "published_year": "연도", "rationale_1": "1회_근거"}
    )

    total = len(merged)
    summary = changed["변경유형"].value_counts().reindex(CHANGE_ORDER).rename("편수").to_frame()
    summary.loc["변경 없음", "편수"] = total - len(changed)
    summary["비율(%)"] = (summary["편수"] / total * 100).round(1)
    summary.loc["합계"] = [total, 100.0]
    summary["편수"] = summary["편수"].astype(int)

    by_field = pd.crosstab(changed["분야"], changed["변경유형"]).reindex(columns=CHANGE_ORDER, fill_value=0)
    by_field["변경 합계"] = by_field.sum(axis=1)
    by_field["분야 전체"] = merged["분야"].value_counts().reindex(by_field.index)
    by_field["변경 비율(%)"] = (by_field["변경 합계"] / by_field["분야 전체"] * 100).round(1)
    by_field = by_field.sort_values("변경 합계", ascending=False)

    group_order = ["ERR", "장", "창", "환경", "장+창", "장+환경", "창+환경", "(과반없음)"]
    transition = pd.crosstab(merged["1회_그룹"], merged["3회_그룹"], margins=True, margins_name="합계")
    rows = [g for g in group_order if g in transition.index] + ["합계"]
    cols = [g for g in group_order if g in transition.columns] + ["합계"]
    transition = transition.reindex(index=rows, columns=cols, fill_value=0)
    transition.index.name = "1회 \\ 3회"

    with pd.ExcelWriter(OUT_XLSX) as writer:
        summary.to_excel(writer, sheet_name="요약")
        by_field.to_excel(writer, sheet_name="분야별")
        transition.to_excel(writer, sheet_name="그룹전이")
        detail.to_excel(writer, sheet_name="변경목록", index=False)

    print(f"비교 대상 {total}편 (초록 없는 논문 제외)\n")
    print(summary.to_string())
    print("\n[그룹 전이] 행=1회, 열=3회")
    print(transition.to_string())
    print("\n[분야별 변경]")
    print(by_field.to_string())
    print(f"\n저장: {OUT_XLSX}")


if __name__ == "__main__":
    main()
