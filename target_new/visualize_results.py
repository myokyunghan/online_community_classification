"""
target_new/classification_results.csv(golden_new 3단계 파이프라인으로 693편을 분류한 결과)를
KCI 사회과학.xlsx의 중분류(분야) 정보와 합쳐서 6개 그래프를 그린다.

1. 전체 시기별(5년 구간) 문헌 수 추이 - 단일 막대그래프 (ERR 제외)
2. 분야별 추이 - 5년 구간 x 분야 누적 막대그래프 (ALLOWED_FIELDS 전부 개별 색, ERR 제외)
3. 분야별 ROLE(역할)/ENV(환경) 비율 - 가로 100% 누적 막대그래프 (ENV 비율 높은 순 정렬)
4. 분야별 개별 코드(10개) 분포 - 히트맵 (분야 x 코드 10개, 3/5번을 코드 단위까지 세분화)
5. 분야별 장(FIELD)/창(WINDOW) 비율 - 가로 100% 누적 막대그래프 (창 비율 높은 순 정렬)
6. 사회학 분야만 - 시기별(5년 구간) 문헌 수 추이
9. 분야 구분 없이 코드(11개)별 문헌 수 - 가로 막대그래프

1·2·6번은 status=ERR(커뮤니티 역할/환경 어디에도 해당 안 됨)인 논문을 빼고 센다. 3·4·5번은
원래부터 status=OK(코드가 실제로 부여된 논문)만 대상으로 한다.

분야(중분류)는 ALLOWED_FIELDS 화이트리스트로 제한한다 - 교육학/관광학/무역학/법학/정책학/
지역학/국제지역개발/행정학처럼 응용·전문분야 성격이 강한 중분류는 통째로 제외하고, "진짜
사회과학"으로 볼 만한 12개 분야만 남긴다(제외 대상은 "기타"로 묶는 게 아니라 아예 분석에서
빠짐).

분야(중분류)는 KCI 원본 엑셀에만 있고 classification_results.csv에는 없어서, 논문ID 기준으로
두 파일을 merge한다. role_codes/env_codes는 한 논문이 여러 개를 가질 수 있어(예: "ROLE_WINDOW_
GENDER, ROLE_WINDOW_HATE") 콤마로 분리해서 코드 등장 횟수 기준으로 집계한다(논문 수 기준이
아님 - 한 논문이 여러 코드에 중복 반영될 수 있다는 뜻).

사용법:
    python3 visualize_results.py
    (target_new/charts/ 아래에 6개 PNG 생성)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

TARGET_DIR = Path(__file__).parent
RESULTS_CSV = TARGET_DIR / "classification_results.csv"
XLSX_PATH = TARGET_DIR / "KCI 사회과학.xlsx"
OUT_DIR = TARGET_DIR / "charts"

# "진짜 사회과학"으로 볼 분야만 추려서 쓴다 (사용자가 직접 고름) - 교육학/관광학/무역학/법학/
# 정책학/지역학/국제·지역개발/행정학처럼 응용·전문분야 성격이 강한 중분류는 제외. 이 목록에
# 없는 분야의 논문은 아예 분석 대상에서 빠진다(기타로 묶지 않고 통째로 제외).
ALLOWED_FIELDS = [
    "경제학", "기타사회과학", "사회과학", "사회과학일반", "사회복지학",
    "사회학", "신문방송학", "심리과학", "인류학", "정치외교학", "지리학", "경영학",
]

TOP_N_FIELDS = len(ALLOWED_FIELDS)  # 화이트리스트라 전부 개별 색으로 표시(기타로 묶을 게 없음)
PERIOD_YEARS = 5  # 시기 구간 폭(년)

# matplotlib 'Blues' 9단계 중 연한 앞부분(거의 흰색)은 빼고 진한 쪽만 써서 더 선명하게 보이게 함
PALETTE = [
    "#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
    "#9ecae1", "#c6dbef", "#c00000",  # 마지막(기타)만 대비되는 색
]

# 히트맵용 - 'Blues' 앞쪽(거의 흰색) 25%를 잘라내고, 나머지를 다시 0~1로 늘려써서 저값도
# 흐릿하지 않고 진하게 보이게 만든 컬러맵
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "deep_blues", plt.get_cmap("Blues")([x / 255 for x in range(64, 256)])
)

ROLE_FIELD_CODES = {"ROLE_FIELD_PUBLICSPHERE", "ROLE_FIELD_SOCIALCAPITAL"}
ROLE_WINDOW_CODES = {
    "ROLE_WINDOW_POLARIZE",
    "ROLE_WINDOW_GENDER",
    "ROLE_WINDOW_HATE",
    "ROLE_WINDOW_MISC",
}

# 원래 11개 코드 정의 순서 (ROLE_FIELD_* -> ROLE_WINDOW_* -> ENV_*) - 4번 히트맵 열 순서와 9번 막대 순서에 공통으로 씀
ALL_CODES = [
    "ROLE_FIELD_PUBLICSPHERE",
    "ROLE_FIELD_SOCIALCAPITAL",
    "ROLE_WINDOW_POLARIZE",
    "ROLE_WINDOW_GENDER",
    "ROLE_WINDOW_HATE",
    "ROLE_WINDOW_MISC",
    "ENV_COMMUNITY_SUBSCRIPTION",
    "ENV_COMMUNITY_GOVERNANCE",
    "ENV_COMMUNITY_DEMOGRAPHIC",
    "ENV_COMMUNITY_LIFECYCLE",
    "ENV_ONLINE_ANNONIMITY",
]


def _setup_korean_font() -> None:
    """macOS 기준 한글 폰트를 잡는다 - 없으면 라벨이 네모(tofu)로 깨진다."""
    for candidate in ("AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic"):
        try:
            plt.rcParams["font.family"] = candidate
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def _colors_for(ordered_cols: list[str]) -> list[str]:
    """분야 개수만큼 색을 만든다. PALETTE(파랑 계열 + 대비색 1개)로 충분하면 그대로 쓰고,
    "기타"가 없거나 분야가 PALETTE보다 많으면(화이트리스트를 12개 다 쓰는 경우 등) 'tab20'
    질적 컬러맵에서 뽑아서 12개 이상도 서로 구분되게 한다."""
    has_etc = ordered_cols[-1] == "기타" if ordered_cols else False
    n_solid = len(ordered_cols) - (1 if has_etc else 0)

    if n_solid <= len(PALETTE) - 1:
        colors = PALETTE[:n_solid]
    else:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i / max(n_solid - 1, 1)) for i in range(n_solid)]

    if has_etc:
        colors = colors + [PALETTE[-1]]
    return colors


def load_merged() -> pd.DataFrame:
    """분류 결과 + 원본 엑셀의 중분류(분야)를 논문ID 기준으로 합친다."""
    results = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig", dtype={"id": str})
    meta = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", dtype={"논문ID": str})
    meta = meta[["논문ID", "중분류"]].rename(columns={"논문ID": "id", "중분류": "field"})

    df = results.merge(meta, on="id", how="left")
    df["field"] = df["field"].fillna("미분류")
    df = df[df["field"].isin(ALLOWED_FIELDS)].copy()

    df["year"] = pd.to_numeric(df["published_year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    period_start = (df["year"] // PERIOD_YEARS * PERIOD_YEARS).astype(int)
    df["decade"] = period_start.astype(str) + "~" + (period_start + PERIOD_YEARS - 1).astype(str)

    return df


def _top_fields(df: pd.DataFrame, n: int = TOP_N_FIELDS) -> list[str]:
    return df["field"].value_counts().head(n).index.tolist()


def _field_bucket_col(df: pd.DataFrame, top_fields: list[str]) -> pd.Series:
    return df["field"].where(df["field"].isin(top_fields), "기타")


def _explode_codes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """role_codes/env_codes 컬럼(콤마로 구분된 문자열)을 코드 하나당 한 행으로 풀어헤친다."""
    sub = df[["field", col]].copy()
    sub[col] = sub[col].fillna("").astype(str)
    sub = sub[sub[col].str.strip() != ""]
    sub[col] = sub[col].str.split(",")
    sub = sub.explode(col)
    sub[col] = sub[col].str.strip()
    sub = sub[sub[col] != ""]
    return sub


def chart1_overall_decade(df: pd.DataFrame) -> None:
    """1. 전체 시기별(5년 구간) 문헌 수 추이 - 단일 막대그래프. ERR(역할/환경 어디에도
    해당 안 됨)은 제외하고 그린다 - 순수하게 "커뮤니티 연구로 인정된" 문헌만 센다."""
    non_err = df[df["status"] != "ERR"]
    counts = non_err["decade"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[1], width=0.6)
    ax.bar_label(bars, padding=3, fontsize=10)

    ax.set_xlabel("발행 시기")
    ax.set_ylabel("논문 수")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_전체_시기별_추이.png", dpi=150)
    plt.close(fig)


def chart_single_field_decade(df: pd.DataFrame, field_name: str, out_name: str) -> None:
    """특정 분야 하나만 추려서 5년 구간별 문헌 수 막대그래프를 그린다 (1번과 동일한 방식,
    대상만 그 분야로 한정). ERR은 마찬가지로 제외."""
    scoped = df[(df["field"] == field_name) & (df["status"] != "ERR")]
    counts = scoped["decade"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[1], width=0.6)
    ax.bar_label(bars, padding=3, fontsize=10)

    ax.set_xlabel("발행 시기")
    ax.set_ylabel("논문 수")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)


def chart7_field_err_rate_bar(df: pd.DataFrame) -> None:
    """7. 분야별 ERR 비율 - 가로 막대그래프. 분야마다 "커뮤니티와 무관하다"고 걸러진(ERR)
    논문이 전체 중 몇 %인지 보여준다 - ERR 비율 높은 분야가 위로 오게 정렬."""
    top_fields = _top_fields(df)
    bucket = _field_bucket_col(df, top_fields)
    total = bucket.value_counts()
    err_total = bucket[df["status"] == "ERR"].value_counts()

    rate = (err_total.reindex(total.index).fillna(0) / total * 100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(rate) + 1.5)))
    bars = ax.barh(rate.index, rate.values, color=PALETTE[1])
    for i, field in enumerate(rate.index):
        ax.text(rate[field] + 1.5, i, f"{rate[field]:.0f}%", va="center", fontsize=9)

    ax.set_yticks(range(len(rate.index)))
    ax.set_yticklabels([f"{field}(n={int(total[field])})" for field in rate.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("ERR 비율(%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "7_분야별_ERR_비율.png", dpi=150)
    plt.close(fig)


def chart8_window_decade(df: pd.DataFrame) -> None:
    """8. 창(WINDOW)에 해당하는 논문 수의 시기별(5년 구간) 추이 - 단일 막대그래프. role_codes에
    ROLE_WINDOW_* 코드가 하나라도 있는 논문만 센다(논문 단위 - 창 코드를 여러 개 가진 논문도
    1편으로만 집계, 4/5번의 "코드 등장 횟수" 집계와는 다르다)."""
    ok = df[df["status"] == "OK"].copy()
    is_window = ok["role_codes"].fillna("").apply(
        lambda s: any(code in ROLE_WINDOW_CODES for code in (c.strip() for c in s.split(",")))
    )
    window_papers = ok[is_window]
    counts = window_papers["decade"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[1], width=0.6)
    ax.bar_label(bars, padding=3, fontsize=10)

    ax.set_xlabel("발행 시기")
    ax.set_ylabel("창(WINDOW) 논문 수")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "8_창_시기별_추이.png", dpi=150)
    plt.close(fig)


def chart9_all_codes_count_bar(df: pd.DataFrame) -> None:
    """9. 분야 구분 없이 코드(11개)별 문헌 수 - 가로 막대그래프. 4번 히트맵을 분야 축 없이
    합친 버전이다. 단, 4번은 코드 등장 횟수 기준이지만 여기서는 논문 단위로 센다 - 한 논문이
    같은 코드를 중복으로 가질 일은 없어서 사실상 같지만, 코드를 여러 개 가진 논문은 각 코드에
    1편씩 반영된다(막대 합 > 전체 논문 수). 0건인 코드도 빠뜨리지 않고 0으로 그린다."""
    ok = df[df["status"] == "OK"].copy()
    all_codes = (ok["role_codes"].fillna("") + "," + ok["env_codes"].fillna("")).apply(
        lambda s: {c.strip() for c in s.split(",") if c.strip()}
    )
    counts = all_codes.explode().dropna().value_counts().reindex(ALL_CODES, fill_value=0)

    # barh는 아래부터 그리므로 뒤집어서 ALL_CODES 순서(ROLE_FIELD -> ROLE_WINDOW -> ENV)가 위에서부터 오게 함
    counts = counts.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(counts) + 1.5)))
    bars = ax.barh(counts.index, counts.values, color=PALETTE[1], height=0.6)
    ax.bar_label(bars, padding=3, fontsize=9)

    ax.set_xlabel("논문 수")
    ax.set_xlim(0, counts.max() * 1.12 if counts.max() else 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "9_전체_코드별_문헌수.png", dpi=150)
    plt.close(fig)


def chart2_field_decade_stacked(df: pd.DataFrame) -> None:
    """2. 분야별 추이 - 5년 구간 x 분야 누적 막대그래프. ERR은 제외."""
    non_err = df[df["status"] != "ERR"]
    top_fields = _top_fields(non_err)
    bucket = _field_bucket_col(non_err, top_fields)
    pivot = pd.crosstab(non_err["decade"], bucket).sort_index()

    # 범례 순서: 상위 분야는 논문 수 많은 순, "기타"는 맨 마지막(대비색)
    ordered_cols = [f for f in top_fields if f in pivot.columns]
    if "기타" in pivot.columns:
        ordered_cols.append("기타")
    pivot = pivot[ordered_cols]

    colors = _colors_for(ordered_cols)

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = pd.Series(0, index=pivot.index)
    for col, color in zip(ordered_cols, colors):
        ax.bar(pivot.index, pivot[col], bottom=bottom, label=col, color=color, width=0.6)
        bottom = bottom + pivot[col]

    ax.set_xlabel("발행 시기")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("논문 수")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_분야별_추이.png", dpi=150)
    plt.close(fig)


def _field_roleenv_table(df: pd.DataFrame) -> pd.DataFrame:
    """분야 x [ROLE(역할), ENV(환경)] 원자료(코드 등장 횟수) 표."""
    top_fields = _top_fields(df)
    ok = df[df["status"] == "OK"].copy()
    ok["field_bucket"] = _field_bucket_col(ok, top_fields)

    role_rows = _explode_codes(ok.assign(field=ok["field_bucket"]), "role_codes")
    env_rows = _explode_codes(ok.assign(field=ok["field_bucket"]), "env_codes")

    role_counts = role_rows.groupby("field").size().rename("ROLE(역할)")
    env_counts = env_rows.groupby("field").size().rename("ENV(환경)")

    order = [f for f in top_fields if f in set(role_counts.index) | set(env_counts.index)]
    if "기타" in (set(role_counts.index) | set(env_counts.index)):
        order.append("기타")

    return pd.concat([role_counts, env_counts], axis=1).reindex(order).fillna(0).astype(int)


def _field_fieldwindow_table(df: pd.DataFrame) -> pd.DataFrame:
    """분야 x [장(FIELD), 창(WINDOW)] 원자료(코드 등장 횟수) 표."""
    top_fields = _top_fields(df)
    ok = df[df["status"] == "OK"].copy()
    ok["field_bucket"] = _field_bucket_col(ok, top_fields)

    role_rows = _explode_codes(ok.assign(field=ok["field_bucket"]), "role_codes")
    role_rows["group"] = role_rows["role_codes"].apply(
        lambda c: "장(FIELD)" if c in ROLE_FIELD_CODES else ("창(WINDOW)" if c in ROLE_WINDOW_CODES else None)
    )
    role_rows = role_rows.dropna(subset=["group"])

    table = pd.crosstab(role_rows["field"], role_rows["group"])
    order = [f for f in top_fields if f in table.index]
    if "기타" in table.index:
        order.append("기타")
    table = table.reindex(order).fillna(0).astype(int)
    for col in ("장(FIELD)", "창(WINDOW)"):
        if col not in table.columns:
            table[col] = 0
    return table[["장(FIELD)", "창(WINDOW)"]]


def _draw_ratio_barh(table: pd.DataFrame, colors: tuple[str, str], sort_col: str, out_path: Path) -> None:
    """분야 x [카테고리 A, 카테고리 B] 원자료(등장 횟수) 표를 가로 100% 누적 막대그래프로
    그린다 - sort_col 비율이 높은 분야가 위로 오게 정렬한다(barh는 아래부터 그리므로 오름차순
    정렬해야 가장 큰 값이 맨 위에 온다). 히트맵(_draw_heatmap)과 동일하게 y축 분야 이름에
    "분야명(n=N)" 형태로 전체 표본 수를 붙인다."""
    col_a, col_b = table.columns
    color_a, color_b = colors

    pct = _row_normalize_pct(table)
    pct = pct.sort_values(sort_col, ascending=True)
    counts = table.loc[pct.index]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(pct) + 1.5)))
    ax.barh(pct.index, pct[col_a], color=color_a, label=col_a)
    ax.barh(pct.index, pct[col_b], left=pct[col_a], color=color_b, label=col_b)

    row_totals = counts[col_a] + counts[col_b]
    ax.set_yticks(range(len(pct.index)))
    ax.set_yticklabels([f"{field}(n={row_totals[field]})" for field in pct.index])

    for i, field in enumerate(pct.index):
        a_pct, b_pct = pct.loc[field, col_a], pct.loc[field, col_b]
        if a_pct > 6:
            ax.text(a_pct / 2, i, f"{a_pct:.0f}%", ha="center", va="center", color="white", fontsize=9)
        if b_pct > 6:
            ax.text(a_pct + b_pct / 2, i, f"{b_pct:.0f}%", ha="center", va="center", color="white", fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_xlabel("비율(%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart3_field_role_env_ratio_bar(df: pd.DataFrame) -> None:
    """3. 분야별 ROLE(역할)/ENV(환경) 비율 - 가로 100% 누적 막대그래프. ENV 비율이 높은
    분야가 위로 오게 정렬(대부분 ROLE이 압도적이라, ENV가 조금이라도 있는 분야를 눈에 띄게)."""
    table = _field_roleenv_table(df)
    _draw_ratio_barh(
        table,
        colors=("#08306b", "#ed7d31"),
        sort_col="ENV(환경)",
        out_path=OUT_DIR / "3_분야별_환경역할_비율_막대.png",
    )


def chart5_field_fieldwindow_ratio_bar(df: pd.DataFrame) -> None:
    """5. 분야별 장(FIELD)/창(WINDOW) 비율 - 가로 100% 누적 막대그래프. 창(WINDOW) 비율이
    높은 분야가 위로 오게 정렬."""
    table = _field_fieldwindow_table(df)
    _draw_ratio_barh(
        table,
        colors=("#08306b", "#ed7d31"),
        sort_col="창(WINDOW)",
        out_path=OUT_DIR / "5_분야별_장창_비율_막대.png",
    )


def chart4_field_all_codes_heatmap(df: pd.DataFrame) -> None:
    """4. 분야별 x 개별 코드(원래 11개 축 중 golden_new가 실제로 판정하는 10개) 등장 횟수
    히트맵 - 3번(ROLE/ENV)과 5번(장/창)의 막대그래프를 코드 단위까지 더 잘게 쪼갠 버전."""
    top_fields = _top_fields(df)
    ok = df[df["status"] == "OK"].copy()
    ok["field_bucket"] = _field_bucket_col(ok, top_fields)

    role_rows = _explode_codes(ok.assign(field=ok["field_bucket"]), "role_codes").rename(
        columns={"role_codes": "code"}
    )
    env_rows = _explode_codes(ok.assign(field=ok["field_bucket"]), "env_codes").rename(
        columns={"env_codes": "code"}
    )
    all_rows = pd.concat([role_rows, env_rows], ignore_index=True)

    table = pd.crosstab(all_rows["field"], all_rows["code"])

    order = [f for f in top_fields if f in table.index]
    if "기타" in table.index:
        order.append("기타")
    table = table.reindex(order).fillna(0).astype(int)

    # 열 순서: ROLE_FIELD_* -> ROLE_WINDOW_* -> ENV_* 순으로 고정 (원래 11개 코드 정의 순서와 맞춤)
    code_order = ALL_CODES
    for c in code_order:
        if c not in table.columns:
            table[c] = 0
    table = table[[c for c in code_order if c in table.columns]]

    _draw_heatmap(
        table,
        title="분야별 코드(10개) 등장 횟수",
        out_path=OUT_DIR / "4_분야별_전체코드_히트맵.png",
        rotate_xlabels=True,
    )


def _row_normalize_pct(table: pd.DataFrame) -> pd.DataFrame:
    """행(분야)별로 합이 100이 되게 정규화한다 - 분야마다 논문 수/코드 총량이 달라서, 원본
    등장 횟수만 보면 큰 분야(경영학 등)가 항상 진하게 보이는 문제를 없앤다. 행 합이 0이면
    (그 분야에 코드가 하나도 없으면) 0으로 둔다(0/0 방지)."""
    row_sums = table.sum(axis=1)
    return table.div(row_sums.where(row_sums != 0, 1), axis=0) * 100


def _draw_heatmap(table: pd.DataFrame, title: str, out_path: Path, rotate_xlabels: bool = False) -> None:
    """table은 원자료(등장 횟수)를 받되, 실제 색상은 행(분야) 기준으로 정규화한 비율로
    칠한다 - 분야별 문헌 수가 크게 달라서 원본 횟수로는 큰 분야만 도드라져 보이기 때문.
    칸 안에는 비율(%)만 적는다(원자료 개수를 칸마다 같이 적으면 오히려 헷갈려서 뺌).
    대신 행(분야)당 전체 원자료 합(n)을 y축 분야 이름에 "분야명(n=N)" 형태로 붙인다 - 표본이 1~2개뿐인
    분야가 100%/0%로 극단적으로 보이는 착시(예: 논문 1편짜리 분야)를 바로 알아채기 위함.

    색상은 PowerNorm(gamma<1)으로 매핑해서 낮은 값(0~20%대)도 색 차이가 잘 보이게 한다.
    범례 상한(vmax)은 최소 50이지만, 실제 데이터 최댓값이 그보다 크면(예: 3번처럼 열이 2개뿐이라
    한쪽이 80~100%까지 올라가는 경우) 그 최댓값까지 늘린다 - 안 그러면 범례는 50까지인데
    칸에는 100%라고 적힌 값이 나와서 범례와 실제 숫자가 안 맞아 보이는 문제가 생긴다."""
    pct = _row_normalize_pct(table)
    vmax = max(50, pct.values.max() if pct.values.size else 0)
    norm = PowerNorm(gamma=0.6, vmin=0, vmax=vmax, clip=True)

    width = max(6, 1.1 * len(table.columns) + 2)
    fig, ax = plt.subplots(figsize=(width, max(4, 0.55 * len(table) + 1.5)))
    im = ax.imshow(pct.values, cmap=HEATMAP_CMAP, aspect="auto", norm=norm)

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, fontsize=9)
    if rotate_xlabels:
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    row_totals = table.sum(axis=1)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{field}(n={row_totals[field]})" for field in table.index], fontsize=10)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            pct_val = pct.values[i, j]
            color = "white" if norm(pct_val) > 0.6 else "black"
            ax.text(j, i, f"{pct_val:.0f}%", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, shrink=0.7, label="분야 내 비율(%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    _setup_korean_font()
    OUT_DIR.mkdir(exist_ok=True)

    df = load_merged()
    print(f"병합된 논문 수(발행년 있는 것만): {len(df)}")

    chart1_overall_decade(df)
    chart2_field_decade_stacked(df)
    chart3_field_role_env_ratio_bar(df)
    chart4_field_all_codes_heatmap(df)
    chart5_field_fieldwindow_ratio_bar(df)
    chart_single_field_decade(df, "사회학", "6_사회학_시기별_추이.png")
    chart7_field_err_rate_bar(df)
    chart8_window_decade(df)
    chart9_all_codes_count_bar(df)

    print(f"완료 - {OUT_DIR} 아래 9개 PNG 생성됨")


if __name__ == "__main__":
    main()
