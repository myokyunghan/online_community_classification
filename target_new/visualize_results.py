"""
target_new/classification_results.csv(golden_new 3단계 파이프라인으로 693편을 분류한 결과)를
KCI 사회과학.xlsx의 중분류(분야) 정보와 합쳐서 6개 그래프를 그린다.

1. 전체 시기별(5년 구간) 문헌 수 추이 - 단일 막대그래프 (ERR 제외)
2. 분야별 추이 - 5년 구간 x 분야 누적 막대그래프 (ALLOWED_FIELDS 전부 개별 색, ERR 제외)
3. 분야별 ROLE(역할)/ENV(환경) 비율 - 가로 100% 누적 막대그래프 (ENV 비율 높은 순 정렬)
3b. 분야별 ROLE(역할)/ENV(환경) 분포 - 양방향(나비형) 가로 막대그래프 (실제 편수, 5b와 같은 형식)
4. 분야별 개별 코드(10개) 분포 - 히트맵 (분야 x 코드 10개, 3/5번을 코드 단위까지 세분화)
4b. 분야별 코드 그룹(ROLE_FIELD/ROLE_WINDOW/ENV_COMMUNITY/ENV_ONLINE) 히트맵
5. 분야별 장(FIELD)/창(WINDOW) 비율 - 가로 100% 누적 막대그래프 (창 비율 높은 순 정렬)
6. 사회학 분야만 - 시기별(5년 구간) 문헌 수 추이
5b. 분야별 장(FIELD)/창(WINDOW) 분포 - 양방향(나비형) 가로 막대그래프 (실제 편수)
9. 분야 구분 없이 코드(11개)별 문헌 수 - 가로 막대그래프
7c. 화이트리스트에서 빠진 분야만 - 분야별 ERR 비율 (7번과 같은 형식)
10. 분야 x 분야 거리 히트맵 - 11개 코드 비율 벡터 사이 Jensen-Shannon 거리 (사회학과 가까운 순)
11. 분야 거리 지도 - 10번 거리를 MDS로 평면에 옮긴 그림 (사회학 강조)

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
    python3 visualize_results.py                       # 1회 분류 결과 -> charts/
    python3 visualize_results.py --results classification_results_runs3.csv --out-dir charts_runs3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

# 막대그래프 공통 색 - 단일 막대(1·6·7·8·9번)와 2색 막대(3·5·5b번)의 남색을 전부 이 색으로 통일한다.
# ACCENT_COLOR는 2색 막대에서 남색과 짝을 이루는 대비색
BAR_COLOR = PALETTE[0]  # "#08306b"
ACCENT_COLOR = "#ed7d31"

# 히트맵용 - 0에서 아주 연한 하늘색으로 시작해 'Blues' 진한 쪽(앞쪽 25%를 잘라낸 부분)으로 이어지는
# 컬러맵. 값이 0인 칸은 따로 흰색으로 비우므로(_draw_heatmap), 시작색을 흰색이 아닌 하늘색으로 둬서
# 작은 값(2~3%)도 흰 칸(0)과 구분되게 한다
# 하늘색 한 색을 Blues 192단계 앞에 그냥 붙이면 전체의 1/193만 차지해서 컬러바 0 근처가 하늘색으로
# 안 보인다 - 그래서 위치를 직접 지정해 0~10% 구간을 하늘색 -> Blues 25% 지점으로 이어지게 한다
_HEATMAP_PALE_END = 0.1
_HEATMAP_BLUES = plt.get_cmap("Blues")([x / 255 for x in range(64, 256)])
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "deep_blues",
    [(0.0, "#e4f1fb")]
    + [
        (_HEATMAP_PALE_END + (1 - _HEATMAP_PALE_END) * i / (len(_HEATMAP_BLUES) - 1), color)
        for i, color in enumerate(_HEATMAP_BLUES)
    ],
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
    """설치된 한글 폰트를 찾아서 쓴다 - 없으면 라벨이 네모(tofu)로 깨진다. rcParams에 이름을
    넣는 것만으로는 폰트가 실제로 있는지 확인되지 않아서, font_manager에 등록된 폰트 이름과
    대조한다. 후보는 macOS -> Linux(나눔고딕) -> Windows(맑은 고딕) 순."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    else:
        print("경고: 한글 폰트를 찾지 못해 그래프의 한글이 깨질 수 있습니다 (INSTALL.md 6장 참고)")
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


def load_merged(excluded: bool = False) -> pd.DataFrame:
    """분류 결과 + 원본 엑셀의 중분류(분야)를 논문ID 기준으로 합친다. 기본은 ALLOWED_FIELDS만
    남기고, excluded=True면 반대로 화이트리스트에서 빠진 분야만 남긴다(7c번 그래프용)."""
    results = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig", dtype={"id": str})
    # 과반 라벨이 없는 논문은 분야·시기별 코드 분석에서만 제외한다. 전체 요약(0번)은 원본 CSV를
    # 별도로 읽어 이 문헌들을 미분류로 포함한다. 리뷰대상 목록은 analyze_runs3.py가 따로 만든다.
    no_label = (
        (results["status"] == "OK")
        & results["role_codes"].fillna("").str.strip().eq("")
        & results["env_codes"].fillna("").str.strip().eq("")
    )
    if no_label.any():
        print(
            f"과반 라벨 없는 논문 {no_label.sum()}편: 0번 요약에는 미분류로 포함, "
            "분야·시기별 그래프에서는 제외 (리뷰 대상)"
        )
    results = results[~no_label]
    meta = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", dtype={"논문ID": str})
    meta = meta[["논문ID", "중분류"]].rename(columns={"논문ID": "id", "중분류": "field"})

    df = results.merge(meta, on="id", how="left")
    df["field"] = df["field"].fillna("미분류")
    in_whitelist = df["field"].isin(ALLOWED_FIELDS)
    df = df[~in_whitelist if excluded else in_whitelist].copy()

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
    bars = ax.bar(counts.index, counts.values, color=BAR_COLOR, width=0.6)
    ax.bar_label(bars, padding=3, fontsize=10)

    ax.set_xlabel("발행 시기")
    ax.set_ylabel("논문 수")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_전체_시기별_추이.png", dpi=150)
    plt.close(fig)


def chart0_classification_summary(df: pd.DataFrame) -> None:
    """원본 분류 CSV 전체를 분모로 ERR/비에러와 비에러 내 코드 비율을 각각 그린다."""
    results = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig", dtype={"id": str})
    results["status"] = results["status"].fillna("").astype(str).str.strip()
    csv_total = len(results)
    if csv_total == 0:
        print("요약 그래프 생략: 추출된 문헌이 없습니다")
        return
    total = csv_total

    is_error = results["status"].eq("ERR")
    non_error = ~is_error
    has_role = results["role_codes"].fillna("").astype(str).str.strip().ne("")
    has_env = results["env_codes"].fillna("").astype(str).str.strip().ne("")
    groups = {
        "역할만": non_error & has_role & ~has_env,
        "환경만": non_error & ~has_role & has_env,
        "역할+환경": non_error & has_role & has_env,
        "미분류": non_error & ~has_role & ~has_env,
    }
    counts = {label: int(mask.sum()) for label, mask in groups.items()}
    err_count = int(is_error.sum())
    non_error_count = int(non_error.sum())
    role_count = int((non_error & has_role).sum())
    env_count = int((non_error & has_env).sum())
    unclassified_count = counts["미분류"]

    overall_labels = ["온라인 커뮤니티와 관련 없음", "온라인 커뮤니티 관련"]
    overall_counts = [err_count, non_error_count]
    overall_colors = [BAR_COLOR, ACCENT_COLOR]
    fig, ax = plt.subplots(figsize=(7, 2.4))
    bars = ax.barh(overall_labels, overall_counts, color=overall_colors, height=0.76)
    for bar, count in zip(bars, overall_counts):
        ax.text(
            bar.get_width() + total * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({count / total:.1%})",
            va="center",
            fontsize=11,
        )
    ax.set_xlim(0, total * 1.16)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.spines[:].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#b8c4d3")
    fig.subplots_adjust(left=0.18, right=0.97, top=0.98, bottom=0.02)
    fig.savefig(OUT_DIR / "0a_전체_ERR_비에러.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    detail_labels = ["미분류", "환경적 특성 조명", "역할적 특성 조명"]
    detail_counts = [unclassified_count, env_count, role_count]
    detail_colors = ["#b8c4d3", ACCENT_COLOR, BAR_COLOR]
    fig, ax = plt.subplots(figsize=(7, 2.55))
    bars = ax.barh(detail_labels, detail_counts, color=detail_colors, height=0.76)
    for bar, count in zip(bars, detail_counts):
        fraction = count / non_error_count if non_error_count else 0
        ax.text(
            bar.get_width() + non_error_count * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({fraction:.1%})",
            va="center",
            fontsize=11,
        )
    ax.set_xlim(0, max(detail_counts, default=0) * 1.22 or 1)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.spines[:].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#b8c4d3")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.98, bottom=0.02)
    fig.savefig(OUT_DIR / "0b_비에러_역할환경_미분류.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_single_field_decade(df: pd.DataFrame, field_name: str, out_name: str) -> None:
    """특정 분야 하나만 추려서 5년 구간별 문헌 수 막대그래프를 그린다 (1번과 동일한 방식,
    대상만 그 분야로 한정). ERR은 마찬가지로 제외."""
    scoped = df[(df["field"] == field_name) & (df["status"] != "ERR")]
    counts = scoped["decade"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=BAR_COLOR, width=0.6)
    ax.bar_label(bars, padding=3, fontsize=10)

    ax.set_xlabel("발행 시기")
    ax.set_ylabel("논문 수")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)


def chart7_field_err_rate_bar(df: pd.DataFrame, out_name: str = "7_분야별_ERR_비율.png") -> None:
    """7. 분야별 ERR 비율 - 가로 막대그래프. 분야마다 "커뮤니티와 무관하다"고 걸러진(ERR)
    논문이 전체 중 몇 %인지 보여준다 - ERR 비율 높은 분야가 위로 오게 정렬. 화이트리스트 밖
    분야(7c번)도 같은 함수로 그린다 - df만 load_merged(excluded=True)로 바꿔서 넘긴다."""
    top_fields = _top_fields(df)
    bucket = _field_bucket_col(df, top_fields)
    total = bucket.value_counts()
    err_total = bucket[df["status"] == "ERR"].value_counts()

    rate = (err_total.reindex(total.index).fillna(0) / total * 100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(rate) + 1.5)))
    bars = ax.barh(rate.index, rate.values, color=BAR_COLOR)
    for i, field in enumerate(rate.index):
        ax.text(rate[field] + 1.5, i, f"{rate[field]:.0f}%", va="center", fontsize=9)

    ax.set_yticks(range(len(rate.index)))
    ax.set_yticklabels([f"{field}(n={int(total[field])})" for field in rate.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("ERR 비율(%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
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
    bars = ax.bar(counts.index, counts.values, color=BAR_COLOR, width=0.6)
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

    # 많이 등장한 코드가 맨 위에 오게 정렬 - barh는 아래부터 그리므로 오름차순으로 정렬해야 함.
    # 동률이면 ALL_CODES 순서를 유지하도록 stable 정렬을 쓴다(뒤집어서 넣고 오름차순 -> 위에서 보면 원래 순서)
    counts = counts.iloc[::-1].sort_values(ascending=True, kind="stable")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(counts) + 1.5)))
    bars = ax.barh(counts.index, counts.values, color=BAR_COLOR, height=0.6)
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


# x축에 분야명을 가로로 놓으면 긴 이름끼리 겹쳐서, 6글자짜리는 두 줄로 나눠 적는다
_FIELD_LABEL_WRAP = {"사회과학일반": "사회과학\n일반", "기타사회과학": "기타\n사회과학"}


def _draw_ratio_bar_ax(ax, table: pd.DataFrame, colors: tuple[str, str], order: list[str]) -> None:
    """분야 x [카테고리 A, 카테고리 B] 원자료(등장 횟수) 표를 주어진 ax에 세로 100% 누적
    막대그래프로 그린다. A가 아래, B가 위에 쌓이고, 분야는 order 순서대로 왼쪽부터 놓인다.
    x축 분야 이름 아래에 "n=N"(그 분야의 두 카테고리 합)을 붙인다. order에 있지만 table에
    없는 분야는 막대 없이 "해당 없음"으로 자리만 남긴다 - 합본 그림에서 위아래 패널의 분야
    위치를 맞추기 위함."""
    col_a, col_b = table.columns
    color_a, color_b = colors

    counts = table.reindex(order).fillna(0).astype(int)
    pct = _row_normalize_pct(counts)
    x = range(len(order))

    ax.bar(x, pct[col_a], color=color_a, label=col_a, width=0.7)
    ax.bar(x, pct[col_b], bottom=pct[col_a], color=color_b, label=col_b, width=0.7)

    for i, field in enumerate(order):
        a_pct, b_pct = pct.loc[field, col_a], pct.loc[field, col_b]
        if a_pct + b_pct == 0:
            ax.text(i, 50, "해당\n없음", ha="center", va="center", color="#999999", fontsize=8)
            continue
        if a_pct >= 8:
            ax.text(i, a_pct / 2, f"{a_pct:.0f}%", ha="center", va="center", color="white", fontsize=8)
        if b_pct >= 8:
            ax.text(i, a_pct + b_pct / 2, f"{b_pct:.0f}%", ha="center", va="center", color="white", fontsize=8)

    row_totals = counts[col_a] + counts[col_b]
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{_FIELD_LABEL_WRAP.get(field, field)}\n(n={row_totals[field]})" for field in order], fontsize=8.5
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("비율(%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)


def _draw_ratio_bar(table: pd.DataFrame, colors: tuple[str, str], sort_col: str, out_path: Path) -> None:
    """단독 그래프용 - sort_col 비율이 높은 분야가 왼쪽에 오게 정렬해서 세로 100% 누적 막대로 그린다."""
    order = _row_normalize_pct(table).sort_values(sort_col, ascending=False, kind="stable").index.tolist()
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(order) + 2), 4.8))
    _draw_ratio_bar_ax(ax, table, colors, order)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart3_field_role_env_ratio_bar(df: pd.DataFrame) -> None:
    """3. 분야별 ROLE(역할)/ENV(환경) 비율 - 세로 100% 누적 막대그래프. ENV 비율이 높은
    분야가 왼쪽에 오게 정렬한다. 표시 n(ROLE/ENV 코드 등장 횟수 합)이 10 미만인 분야는 뺀다."""
    table = _field_roleenv_table(df)
    table = table.loc[table.sum(axis=1) >= 10]
    _draw_ratio_bar(
        table,
        colors=(BAR_COLOR, ACCENT_COLOR),
        sort_col="ENV(환경)",
        out_path=OUT_DIR / "3_분야별_환경역할_비율_막대.png",
    )


def chart3b_field_roleenv_diverging(df: pd.DataFrame) -> None:
    """3b. 분야별 역할(ROLE)/환경(ENV) 분포 - 양방향(나비형) 가로 막대그래프. 5b(장/창)와 같은
    형식으로, 그 한 단계 위 축인 역할(ROLE 6개 코드 중 하나라도)/환경(ENV 4개 코드 중 하나라도)을
    나눠 그린다. 3번(100% 비율 막대)과 달리 실제 편수를 그대로 보여줘서 분야 간 규모 차이가
    같이 보인다 - 3번에서는 심리과학처럼 표본이 작은 분야도 다른 분야와 똑같이 100% 기준
    막대로 그려져 착시가 생기는데, 여기서는 막대 길이 자체가 짧게 나와 바로 드러난다.

    3번은 코드 등장 횟수 기준이지만 여기서는 5b처럼 논문 단위로 센다 - 역할 코드가 하나라도
    있으면 역할 1편, 환경 코드가 하나라도 있으면 환경 1편. 둘 다 가진 논문은 양쪽에 1편씩
    들어가서, 왼쪽 "N편"(역할 또는 환경이 있는 논문 수)이 막대 두 개의 합보다 작을 수 있다.
    N편이 큰 분야가 위로 온다. N편이 10 미만인 분야는 뺀다(5b는 아직 이 필터가 없음)."""
    ok = df[df["status"] == "OK"].copy()
    ok["has_role"] = ok["role_codes"].fillna("").str.strip() != ""
    ok["has_env"] = ok["env_codes"].fillna("").str.strip() != ""
    ok = ok[ok["has_role"] | ok["has_env"]]

    table = ok.groupby("field").agg(n=("id", "size"), role_n=("has_role", "sum"), env_n=("has_env", "sum"))
    table = table[table["n"] >= 10]
    # barh는 아래부터 그리므로 오름차순 정렬해야 N편이 큰 분야가 맨 위에 온다
    table = table.sort_values("n", ascending=True, kind="stable")

    role_color, env_color = BAR_COLOR, ACCENT_COLOR
    y = range(len(table))
    xmax = max(table["role_n"].max(), table["env_n"].max()) * 1.08

    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(table) + 1.5)))
    ax.barh(y, -table["role_n"], color=role_color, height=0.55)
    ax.barh(y, table["env_n"], color=env_color, height=0.55)
    ax.axvline(0, color="#d0d0d0", linewidth=1, zorder=0)

    # 막대 안쪽 끝에 흰 글씨로 편수 표시 - 막대가 너무 짧으면(축 폭의 6% 미만) 바깥쪽에 검은 글씨로
    min_inside = xmax * 0.06
    for i, (r_n, e_n) in enumerate(zip(table["role_n"], table["env_n"])):
        if r_n >= min_inside:
            ax.text(-r_n + xmax * 0.01, i, str(r_n), ha="left", va="center", color="white", fontsize=10, fontweight="bold")
        elif r_n > 0:
            ax.text(-r_n - xmax * 0.01, i, str(r_n), ha="right", va="center", color="black", fontsize=10, fontweight="bold")
        if e_n >= min_inside:
            ax.text(e_n - xmax * 0.01, i, str(e_n), ha="right", va="center", color="white", fontsize=10, fontweight="bold")
        elif e_n > 0:
            ax.text(e_n + xmax * 0.01, i, str(e_n), ha="left", va="center", color="black", fontsize=10, fontweight="bold")

    # y축 라벨 대신 왼쪽 여백에 분야명만 적는다 (N편 표시는 뺌)
    ax.set_yticks([])
    for i, field in enumerate(table.index):
        ax.text(-0.22, i, field, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=11, fontweight="bold")

    ax.text(-xmax * 0.02, len(table) - 0.35, "◀ 역할 ROLE", ha="right", va="bottom", color=role_color, fontsize=10)
    ax.text(xmax * 0.02, len(table) - 0.35, "환경 ENV ▶", ha="left", va="bottom", color=env_color, fontsize=10)

    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-0.6, len(table) - 0.1)
    ax.set_xticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.03)
    fig.savefig(OUT_DIR / "3b_분야별_역할환경_분포_양방향.png", dpi=150)
    plt.close(fig)


def chart5_field_fieldwindow_ratio_bar(df: pd.DataFrame) -> None:
    """5. 분야별 장(FIELD)/창(WINDOW) 비율 - 세로 100% 누적 막대그래프. 창(WINDOW) 비율이
    높은 분야가 왼쪽에 오게 정렬."""
    _draw_ratio_bar(
        _field_fieldwindow_table(df),
        colors=(BAR_COLOR, ACCENT_COLOR),
        sort_col="창(WINDOW)",
        out_path=OUT_DIR / "5_분야별_장창_비율_막대.png",
    )


def chart3_5_combined(df: pd.DataFrame) -> None:
    """3+5. 논문 게재용 합본 - (a) ROLE/ENV 비율, (b) 장/창 비율을 위아래 두 패널로 넣는다.
    두 패널에서 같은 분야를 쉽게 찾아 비교할 수 있게 분야 순서를 통일한다((a) 패널의 n이 큰
    분야부터 왼쪽). 단독 그래프(3·5번)처럼 비율로 정렬하면 패널마다 순서가 달라져서 비교가
    어렵다. (b)에 없는 분야(ROLE 코드 없이 ENV만 있는 분야)는 (b)에서 막대 없이 "해당 없음"으로
    자리만 남겨서, 위아래 패널의 같은 분야가 같은 세로줄에 오게 한다.

    인쇄용이라 PNG는 300dpi로, 확대해도 깨지지 않게 PDF(벡터)도 같이 저장한다."""
    roleenv = _field_roleenv_table(df)
    fieldwindow = _field_fieldwindow_table(df)
    # 그림에 적힌 (a) 패널의 n(코드 수)이 큰 분야부터 왼쪽에 놓는다 - 논문 편수로 정렬하면 적힌 n과
    # 순서가 어긋나 보여서(예: 신문방송학 n=63이 경영학 n=62 뒤에 옴) 그림에 보이는 숫자 기준으로 맞춤
    order = roleenv.sum(axis=1).sort_values(ascending=False, kind="stable").index.tolist()

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8.5, 8.6))
    _draw_ratio_bar_ax(ax_a, roleenv, (BAR_COLOR, ACCENT_COLOR), order)
    _draw_ratio_bar_ax(ax_b, fieldwindow, (BAR_COLOR, ACCENT_COLOR), order)
    for ax, label in ((ax_a, "(a) 역할(ROLE)과 환경(ENV)"), (ax_b, "(b) 장(FIELD)과 창(WINDOW)")):
        ax.set_title(label, loc="left", fontsize=11, pad=22)

    fig.tight_layout(h_pad=2.5)
    out_stem = OUT_DIR / "3_5_합본_역할환경_장창_비율"
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def chart5b_field_fieldwindow_diverging(df: pd.DataFrame) -> None:
    """5b. 분야별 장(FIELD)/창(WINDOW) 분포 - 양방향(나비형) 가로 막대그래프. 가운데 축을
    기준으로 장은 왼쪽(파랑), 창은 오른쪽(주황)으로 뻗는다. 5번(100% 비율)과 달리 실제 편수를
    그대로 보여줘서 분야 간 규모 차이도 같이 보인다.

    5번은 코드 등장 횟수 기준이지만 여기서는 논문 단위로 센다 - 장 코드가 하나라도 있으면 장 1편,
    창 코드가 하나라도 있으면 창 1편. 장·창을 둘 다 가진 논문은 양쪽에 1편씩 들어가서, 왼쪽
    "N편"(장 또는 창이 있는 논문 수)이 막대 두 개의 합보다 작을 수 있다. N편이 큰 분야가 위로 온다."""
    ok = df[df["status"] == "OK"].copy()
    code_sets = ok["role_codes"].fillna("").apply(lambda s: {c.strip() for c in s.split(",") if c.strip()})
    ok["has_field"] = code_sets.apply(lambda codes: bool(codes & ROLE_FIELD_CODES))
    ok["has_window"] = code_sets.apply(lambda codes: bool(codes & ROLE_WINDOW_CODES))
    ok = ok[ok["has_field"] | ok["has_window"]]

    table = ok.groupby("field").agg(
        n=("id", "size"), field_n=("has_field", "sum"), window_n=("has_window", "sum")
    )
    # barh는 아래부터 그리므로 오름차순 정렬해야 N편이 큰 분야가 맨 위에 온다
    table = table.sort_values("n", ascending=True, kind="stable")

    field_color, window_color = BAR_COLOR, ACCENT_COLOR
    y = range(len(table))
    xmax = max(table["field_n"].max(), table["window_n"].max()) * 1.08

    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(table) + 1.5)))
    ax.barh(y, -table["field_n"], color=field_color, height=0.55)
    ax.barh(y, table["window_n"], color=window_color, height=0.55)
    ax.axvline(0, color="#d0d0d0", linewidth=1, zorder=0)

    # 막대 안쪽 끝에 흰 글씨로 편수 표시 - 막대가 너무 짧으면(축 폭의 6% 미만) 바깥쪽에 검은 글씨로
    min_inside = xmax * 0.06
    for i, (f_n, w_n) in enumerate(zip(table["field_n"], table["window_n"])):
        if f_n >= min_inside:
            ax.text(-f_n + xmax * 0.01, i, str(f_n), ha="left", va="center", color="white", fontsize=10, fontweight="bold")
        elif f_n > 0:
            ax.text(-f_n - xmax * 0.01, i, str(f_n), ha="right", va="center", color="black", fontsize=10, fontweight="bold")
        if w_n >= min_inside:
            ax.text(w_n - xmax * 0.01, i, str(w_n), ha="right", va="center", color="white", fontsize=10, fontweight="bold")
        elif w_n > 0:
            ax.text(w_n + xmax * 0.01, i, str(w_n), ha="left", va="center", color="black", fontsize=10, fontweight="bold")

    # y축 라벨 대신 왼쪽 여백에 "분야명   N편" 두 칸으로 직접 적는다(스크린샷 형태)
    ax.set_yticks([])
    for i, (field, n) in enumerate(zip(table.index, table["n"])):
        ax.text(-0.22, i, field, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=11, fontweight="bold")
        ax.text(-0.02, i, f"{n}편", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10, color="#888888")

    ax.text(-xmax * 0.02, len(table) - 0.35, "◀ 장 FIELD", ha="right", va="bottom", color=field_color, fontsize=10)
    ax.text(xmax * 0.02, len(table) - 0.35, "창 WINDOW ▶", ha="left", va="bottom", color=window_color, fontsize=10)

    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-0.6, len(table) - 0.1)
    ax.set_xticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.03)
    fig.savefig(OUT_DIR / "5b_분야별_장창_분포_양방향.png", dpi=150)
    plt.close(fig)


def _field_code_table(df: pd.DataFrame, min_papers: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """분야 x 개별 코드(11개) 등장 횟수 표와, 분야별 문헌 수(status=ERR/NO_ABSTRACT 제외)를
    만든다. 4번(11개 코드)과 4b번(2단계 그룹 4개) 히트맵이 같은 집계를 쓰도록 공통으로 뺐다.

    분야를 히트맵에 포함시킬지, 그리고 y축에 "(n=N)"으로 표시할 값 둘 다 status=OK인 문헌 수를
    기준으로 한다 - ERR(온라인 커뮤니티 역할/환경 어디에도 해당 안 됨)이나 NO_ABSTRACT(초록이
    없어 애초에 분류를 못 한 논문)까지 포함해서 세면, 실제로는 role/env 코드가 하나도 없는
    분야가 "문헌이 10편 넘는다"는 이유로 포함되거나 n이 실제 코드 분포와 무관하게 커 보이는
    문제가 있었다. status=OK는 이 데이터에서 전부 role/env 코드가 최소 1개는 있으므로(과반
    라벨 없는 논문은 load_merged가 이미 제외), y축 n이 곧 그 분야의 "역할 또는 환경으로
    분류된 논문 수"와 일치한다."""
    top_fields = _top_fields(df)
    ok = df[df["status"] == "OK"].copy()
    ok["field_bucket"] = _field_bucket_col(ok, top_fields)
    field_totals = ok["field_bucket"].value_counts()  # 포함 여부 판단 + y축 n 표시 기준

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

    # 히트맵 열 순서는 창(ROLE_WINDOW_*) -> 장(ROLE_FIELD_*) -> ENV_* (4b번 그룹 순서와 맞춤)
    heatmap_order = [c for c in ALL_CODES if c.startswith("ROLE_WINDOW")] + [
        c for c in ALL_CODES if not c.startswith("ROLE_WINDOW")
    ]
    table = table.reindex(columns=heatmap_order, fill_value=0)

    # min_papers: 히트맵에 개별 행으로 넣을 최소 문헌 수. 4b번은 작은 분야를 "그 외"로 합쳐서
    # 보여주므로 0으로 불러 모든 분야를 받은 뒤 직접 합친다
    included_fields = field_totals[field_totals >= min_papers].index
    table = table.loc[table.index.intersection(included_fields)]
    field_counts = field_totals.reindex(table.index).fillna(0).astype(int)
    return table, field_counts


def chart4_field_all_codes_heatmap(df: pd.DataFrame) -> None:
    """4. 분야별 x 개별 코드(11개) 등장 횟수 히트맵 - 3번(ROLE/ENV)과 5번(장/창)의 막대그래프를
    코드 단위까지 더 잘게 쪼갠 버전."""
    table, field_counts = _field_code_table(df)
    _draw_heatmap(
        table,
        title="분야별 코드(11개) 등장 횟수",
        out_path=OUT_DIR / "4_분야별_전체코드_히트맵.png",
        rotate_xlabels=True,
        row_n=field_counts,
    )


# 코드 이름 앞 두 단계로 묶은 그룹 - 4b번 히트맵 x축 순서
CODE_GROUPS = ["ROLE_WINDOW", "ROLE_FIELD", "ENV_COMMUNITY", "ENV_ONLINE"]  # 히트맵 열 순서: 창 -> 장 -> 환경

# 4b번 히트맵 x축 틱 라벨 - 코드명 그대로가 아니라 "역할적/환경적 측면"이라는 상위 축 이름을
# 위 줄에, 장/창/커뮤니티/온라인이라는 하위 구분을 아래 줄에 적는다
CODE_GROUP_LABELS = {
    "ROLE_FIELD": "역할적 측면\n장",
    "ROLE_WINDOW": "역할적 측면\n창",
    "ENV_COMMUNITY": "환경적 측면\n커뮤니티",
    "ENV_ONLINE": "환경적 측면\n온라인",
}


# 사람이 직접 판정한 사회학 논문(golden). 첫 줄에 논문 정보 + 첫 라벨, 2·3번째 라벨은 그 아래 줄에
# 논문명 없이 이어진다. 4b번 히트맵 맨 위에 "사회학(golden)" 행으로 따로 붙인다
GOLDEN_XLSX = TARGET_DIR.parent / "full_golden.xlsx"
GOLDEN_ROW_LABEL = "사회학"  # 패널 제목에 "사람 판정 (golden)"이 있어 행 이름엔 분야만


def _load_golden() -> pd.DataFrame | None:
    """golden 엑셀을 읽어 중복 논문을 뺀 라벨 행(한 줄 = 라벨 하나)을 돌려준다. paper 열이 논문 번호,
    연도 열은 그 논문의 첫 줄 값으로 채운다. 파일이 없으면 None. 4b 히트맵과 12번 연도 그래프가 같은
    56편을 쓰도록 공통으로 뺐다."""
    if not GOLDEN_XLSX.exists():
        print(f"golden 파일 없음 - golden 기반 그래프(4b 위 패널, 12번)를 건너뜀: {GOLDEN_XLSX}")
        return None
    golden = pd.read_excel(GOLDEN_XLSX)
    golden["paper"] = golden["논문명"].notna().cumsum()  # 논문명이 있는 줄마다 새 논문 시작
    golden["title"] = golden["논문명"].ffill().str.replace(r"\s+", "", regex=True)
    golden["연도"] = golden["연도"].ffill()

    # 같은 논문이 2~3번 중복으로 들어 있는 경우가 있다(2026-10-01 기준 73건 중 14편이 중복, 라벨도
    # 동일). 그대로 세면 라벨이 많은 논문이 2~3배로 집계되므로 제목(공백 제거) 기준 첫 번째만 쓴다
    first_paper = golden.groupby("title")["paper"].transform("min")
    deduped = golden[golden["paper"] == first_paper]
    n_dup = golden["paper"].nunique() - deduped["paper"].nunique()
    if n_dup:
        print(f"golden: 중복 논문 {n_dup}건 제외 -> {deduped['paper'].nunique()}편")
    return deduped


def _golden_group_row() -> tuple[pd.DataFrame, pd.Series] | None:
    """golden 라벨을 코드 그룹(4개) 등장 횟수 1행 표와 논문 수로 만든다. 파일이 없으면 None."""
    deduped = _load_golden()
    if deduped is None:
        return None
    labels = deduped["Golden label"].dropna().str.strip()
    groups = labels.map(lambda code: "_".join(code.split("_")[:2]))
    row = groups.value_counts().reindex(CODE_GROUPS, fill_value=0)
    table = row.rename(GOLDEN_ROW_LABEL).to_frame().T
    n = pd.Series({GOLDEN_ROW_LABEL: int(deduped["paper"].nunique())})
    return table, n


# 12번 그래프의 출간 연도 구간 - 5년 단위, 2004년부터(golden 최초 논문 연도)
GOLDEN_PERIOD_BINS = [(2004, 2008), (2009, 2013), (2014, 2018), (2019, 2023), (2024, 2028)]


def chart12_golden_sociology_period() -> None:
    """12. 사람이 판정한 사회학 논문(golden, 중복 제거) 출간 연도별 문헌 수 - 단일 막대그래프.
    4b 히트맵 위 패널과 같은 논문 집합이다. 스타일은 1·6·8번 시기별 막대그래프와 같다."""
    deduped = _load_golden()
    if deduped is None:
        return
    years = deduped.groupby("paper")["연도"].first().astype(int)
    labels = [f"{start}~{end}" for start, end in GOLDEN_PERIOD_BINS]
    counts = pd.Series(
        [int(((years >= start) & (years <= end)).sum()) for start, end in GOLDEN_PERIOD_BINS], index=labels
    )
    outside = len(years) - counts.sum()
    if outside:
        print(f"경고: 12번 그래프 구간({GOLDEN_PERIOD_BINS[0][0]}~{GOLDEN_PERIOD_BINS[-1][1]}) 밖 논문 {outside}편은 빠짐")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=BAR_COLOR, width=0.6)
    ax.bar_label(bars, padding=3, fontsize=13)

    # 논문 그림으로 쓸 때 잘 보이게 축 제목·눈금 글씨를 기본(10)보다 키운다
    ax.set_xlabel("발행 시기", fontsize=14)
    ax.set_ylabel("논문 수", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "12_사회학_사람판정_시기별_문헌수.png", dpi=150)
    plt.close(fig)


# 4b번 히트맵에서 따로 보여줄 주요 분야(이 순서대로 위에서부터) - 나머지는 "그 외" 한 행으로 합친다
HEATMAP_MAIN_FIELDS = ["경영학", "신문방송학", "사회과학일반", "사회학"]
OTHER_FIELDS_LABEL = "그 외"


def chart4b_field_code_group_heatmap(df: pd.DataFrame) -> None:
    """4b. 분야별 x 코드 그룹(2단계) 히트맵 - 11개 코드를 이름 앞 두 단계(ROLE_FIELD / ROLE_WINDOW /
    ENV_COMMUNITY / ENV_ONLINE)로 묶어 합친다. 4번과 같은 집계를 열만 합친 것이라, 각 칸은 그 분야
    코드 중 해당 그룹이 차지하는 비율이고 y축 n은 그 분야의 role/env 코드 부여 문헌 수(포함 기준과 동일)다."""
    table, field_counts = _field_code_table(df, min_papers=0)  # 작은 분야도 받아서 "그 외"로 합친다
    grouped = table.T.groupby(lambda code: "_".join(code.split("_")[:2])).sum().T
    grouped = grouped.reindex(columns=CODE_GROUPS, fill_value=0)
    # 열이 4개뿐이라 칸이 좁아서 한 줄로 쓰면 이웃 라벨과 겹친다 - 위/아래 두 줄로 나눠 표시
    grouped.columns = [CODE_GROUP_LABELS[group] for group in grouped.columns]

    # golden(사람 판정 사회학)이 있으면 위 패널이 사회학을 대표하므로, 아래 모델 패널에서는 사회학을
    # 아예 뺀다("그 외"에도 넣지 않음) - 같은 분야가 위아래 두 번 나오지 않게
    golden = _golden_group_row()
    if golden is not None:
        grouped = grouped.drop(index=REFERENCE_FIELD, errors="ignore")

    # 주요 분야는 그대로, 나머지는 코드 수와 문헌 수를 더해 "그 외" 한 행으로 합친다. 작은 분야는
    # 문헌이 몇 편뿐이라 비율이 극단적으로 튀어서(예: 인류학 1편 -> 100%) 따로 보여주면 오히려 헷갈린다
    main = [f for f in HEATMAP_MAIN_FIELDS if f in grouped.index]
    others = [f for f in grouped.index if f not in main]
    rows = grouped.loc[main]
    row_n = field_counts.reindex(main).fillna(0).astype(int)
    if others:
        rows = pd.concat([rows, grouped.loc[others].sum().rename(OTHER_FIELDS_LABEL).to_frame().T])
        row_n[OTHER_FIELDS_LABEL] = int(field_counts.reindex(others).fillna(0).sum())
        print(f"4b 히트맵 '{OTHER_FIELDS_LABEL}' 행에 합친 분야: {', '.join(others)}")

    reference = None
    if golden is not None:
        golden_table, golden_n = golden
        golden_table.columns = [CODE_GROUP_LABELS[group] for group in golden_table.columns]
        reference = (golden_table, golden_n, "사람 판정", "LLM 분류기 판정")

    _draw_heatmap(
        rows,
        title="분야별 코드 그룹(4개) 등장 횟수",
        out_path=OUT_DIR / "4b_분야별_코드그룹_히트맵.png",
        row_n=row_n,
        reference=reference,
    )


def _row_normalize_pct(table: pd.DataFrame) -> pd.DataFrame:
    """행(분야)별로 합이 100이 되게 정규화한다 - 분야마다 논문 수/코드 총량이 달라서, 원본
    등장 횟수만 보면 큰 분야(경영학 등)가 항상 진하게 보이는 문제를 없앤다. 행 합이 0이면
    (그 분야에 코드가 하나도 없으면) 0으로 둔다(0/0 방지)."""
    row_sums = table.sum(axis=1)
    return table.div(row_sums.where(row_sums != 0, 1), axis=0) * 100


def _heatmap_panel(
    ax,
    table: pd.DataFrame,
    pct: pd.DataFrame,
    row_totals: pd.Series,
    norm,
    cmap,
    show_xticks: bool = True,
    rotate_xlabels: bool = False,
):
    """히트맵 한 패널을 ax에 그린다 - 0인 칸은 흰색 + 회색 "0%", 나머지는 비율(%)을 적는다."""
    masked = np.ma.masked_where(table.values == 0, pct.values)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", norm=norm)

    ax.set_xticks(range(len(table.columns)))
    if show_xticks:
        ax.set_xticklabels(table.columns, fontsize=9)
        if rotate_xlabels:
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{field}(n={row_totals[field]})" for field in table.index], fontsize=10)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            pct_val = pct.values[i, j]
            if table.values[i, j] == 0:
                ax.text(j, i, "0%", ha="center", va="center", color="#c8c8c8", fontsize=9)
                continue
            color = "white" if norm(pct_val) > 0.6 else "black"
            ax.text(j, i, f"{pct_val:.0f}%", ha="center", va="center", color=color, fontsize=10)
    return im


def _draw_heatmap(
    table: pd.DataFrame,
    title: str,
    out_path: Path,
    rotate_xlabels: bool = False,
    row_n: pd.Series | None = None,
    reference: tuple[pd.DataFrame, pd.Series, str, str] | None = None,
) -> None:
    """table은 원자료(등장 횟수)를 받되, 실제 색상은 행(분야) 기준으로 정규화한 비율로
    칠한다 - 분야별 문헌 수가 크게 달라서 원본 횟수로는 큰 분야만 도드라져 보이기 때문.
    칸 안에는 비율(%)만 적는다(원자료 개수를 칸마다 같이 적으면 오히려 헷갈려서 뺌).
    대신 행(분야)당 전체 원자료 합(n)을 y축 분야 이름에 "분야명(n=N)" 형태로 붙인다 - 표본이 1~2개뿐인
    분야가 100%/0%로 극단적으로 보이는 착시(예: 논문 1편짜리 분야)를 바로 알아채기 위함.
    row_n을 주면 원자료 합 대신 그 값(예: 분야별 논문 편수)을 n으로 쓴다.

    reference=(표, n, 위 패널 제목, 아래 패널 제목)을 주면 그 표를 위쪽에 간격을 두고 별도 패널로
    그린다(4b번의 사람 판정 golden 행용). 두 패널은 같은 색 기준(norm)과 컬러바 하나를 공유해서
    색을 그대로 비교할 수 있다.

    색상은 PowerNorm(gamma<1)으로 매핑해서 낮은 값(0~20%대)도 색 차이가 잘 보이게 한다.
    범례 상한(vmax)은 최소 50이지만, 실제 데이터 최댓값이 그보다 크면(예: 3번처럼 열이 2개뿐이라
    한쪽이 80~100%까지 올라가는 경우) 그 최댓값까지 늘린다 - 안 그러면 범례는 50까지인데
    칸에는 100%라고 적힌 값이 나와서 범례와 실제 숫자가 안 맞아 보이는 문제가 생긴다."""
    pct = _row_normalize_pct(table)
    row_totals = row_n.reindex(table.index).fillna(0).astype(int) if row_n is not None else table.sum(axis=1)
    ref_pct = _row_normalize_pct(reference[0]) if reference is not None else None

    # 상한을 10 단위로 올림한다 - 최댓값(예: 62%) 그대로 두면 matplotlib이 상한을 넘는 눈금(70)을
    # 컬러바 끝에 붙여 60과 겹쳐 보인다. 눈금도 상한까지 직접 지정한다(아래 cbar.set_ticks).
    # reference가 있으면 두 패널 중 큰 값 기준으로 잡아 같은 색이 같은 비율을 뜻하게 한다
    max_pct = max(pct.values.max() if pct.values.size else 0, ref_pct.values.max() if ref_pct is not None else 0)
    vmax = max(50, int(np.ceil(max_pct / 10) * 10))
    norm = PowerNorm(gamma=0.6, vmin=0, vmax=vmax, clip=True)
    # 0인 칸(코드가 한 번도 안 나온 칸)은 색을 칠하지 않고 흰색으로 둔다 - 값이 있는 칸만 눈에 띄게
    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad("white")

    width = max(6, 1.1 * len(table.columns) + 2)
    if reference is None:
        fig, ax = plt.subplots(figsize=(width, max(4, 0.55 * len(table) + 1.5)))
        im = _heatmap_panel(ax, table, pct, row_totals, norm, cmap, rotate_xlabels=rotate_xlabels)
        cbar_axes = ax
        n_rows = len(table.index)
    else:
        ref_table, ref_n, ref_title, main_title = reference
        ref_totals = ref_n.reindex(ref_table.index).fillna(0).astype(int)
        n_ref, n_main = len(ref_table), len(table)
        fig, (ax_ref, ax) = plt.subplots(
            2, 1, figsize=(width, 0.55 * (n_ref + n_main) + 2.4),
            gridspec_kw={"height_ratios": [n_ref, n_main]}, layout="constrained",
        )
        fig.get_layout_engine().set(hspace=0.08)  # 두 패널 사이 간격 - golden과 모델 결과를 구분
        _heatmap_panel(ax_ref, ref_table, ref_pct, ref_totals, norm, cmap, show_xticks=False)
        im = _heatmap_panel(ax, table, pct, row_totals, norm, cmap, rotate_xlabels=rotate_xlabels)
        ax_ref.set_title(ref_title, loc="left", fontsize=9, color="#555555")
        ax.set_title(main_title, loc="left", fontsize=9, color="#555555")
        cbar_axes = [ax_ref, ax]
        n_rows = n_ref + n_main

    cbar = fig.colorbar(im, ax=cbar_axes, shrink=0.7, label="분야 내 비율(%)")
    # 행이 적은 히트맵은 컬러바가 짧아서 10 단위 눈금이 위쪽에서 겹친다 - 그때는 20 단위로
    tick_step = 10 if vmax <= 70 and n_rows >= 5 else 20
    cbar.set_ticks(list(range(0, vmax + 1, tick_step)))
    cbar.outline.set_edgecolor("#999999")
    if reference is None:
        fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart3_5_compact(df: pd.DataFrame) -> None:
    """3+5 압축본 - 논문 본문 폭(약 16cm)용. ROLE 코드는 전부 장(FIELD) 아니면 창(WINDOW)이라,
    분야마다 막대 하나를 [장 / 창 / 환경(ENV)] 세 구간으로 나누면 합본의 (a)·(b) 두 패널 정보가
    한 패널에 다 들어간다. 비율의 분모는 그 분야의 전체 코드 수(n)라서, 장·창 비율이 (b) 패널
    (ROLE 코드만 분모)보다 ENV 몫만큼 조금 작게 나온다."""
    roleenv = _field_roleenv_table(df)
    fieldwindow = _field_fieldwindow_table(df)
    table = pd.concat([fieldwindow, roleenv[["ENV(환경)"]]], axis=1).fillna(0).astype(int)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    pct = _row_normalize_pct(table)

    colors = {"장(FIELD)": BAR_COLOR, "창(WINDOW)": ACCENT_COLOR, "ENV(환경)": "#9fb4cc"}
    text_colors = {"장(FIELD)": "white", "창(WINDOW)": "white", "ENV(환경)": "#1f2d3d"}

    with plt.rc_context({"font.size": 8}):
        fig, ax = plt.subplots(figsize=(6.3, 3.0))  # 16cm x 7.6cm
        x = range(len(table))
        bottom = pd.Series(0.0, index=table.index)
        for col in table.columns:
            ax.bar(x, pct[col], bottom=bottom, color=colors[col], label=col, width=0.72, edgecolor="white", linewidth=0.6)
            for i, field in enumerate(table.index):
                if pct.loc[field, col] >= 10:
                    ax.text(i, bottom[field] + pct.loc[field, col] / 2, f"{pct.loc[field, col]:.0f}",
                            ha="center", va="center", color=text_colors[col], fontsize=7)
            bottom = bottom + pct[col]

        n = table.sum(axis=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{_FIELD_LABEL_WRAP.get(f, f)}\n(n={n[f]})" for f in table.index], fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel("비율(%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.5,
                  handlelength=1.2, columnspacing=1.5)
        fig.tight_layout(pad=0.4)
        out_stem = OUT_DIR / "3_5_압축_장창환경_비율"
        fig.savefig(out_stem.with_suffix(".png"), dpi=600)
        fig.savefig(out_stem.with_suffix(".pdf"))
        plt.close(fig)


REFERENCE_FIELD = "사회학"  # 분야 간 거리(10·11번)의 기준 분야 - 연구의 중심 분야
MIN_CODES_FOR_DISTANCE = 10  # 코드가 이보다 적은 분야는 비율 벡터가 불안정해서 거리 분석에서 뺀다


def _field_code_share(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """분야마다 11개 코드의 비율 벡터(행 합=1)와 코드 수를 만든다. 원래 개수 대신 비율을 쓰는
    이유: 개수 그대로 거리를 재면 코드 구성이 비슷해도 분야 크기(경영학 62 vs 심리과학 4)
    차이 때문에 멀게 나온다. MIN_CODES_FOR_DISTANCE 미만인 분야는 뺀다."""
    ok = df[df["status"] == "OK"]
    rows = pd.concat(
        [
            _explode_codes(ok, "role_codes").rename(columns={"role_codes": "code"}),
            _explode_codes(ok, "env_codes").rename(columns={"env_codes": "code"}),
        ]
    )
    counts = pd.crosstab(rows["field"], rows["code"]).reindex(columns=ALL_CODES, fill_value=0)
    n_codes = counts.sum(axis=1)
    counts = counts[n_codes >= MIN_CODES_FOR_DISTANCE]
    return counts.div(counts.sum(axis=1), axis=0), n_codes[counts.index]


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """두 확률분포 사이 Jensen-Shannon 거리(log2 기준, 0=같음 ~ 1=완전히 다름). scipy가 없어 직접 구현."""
    m = (p + q) / 2

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(np.sqrt((_kl(p, m) + _kl(q, m)) / 2))


def _field_distance_matrix(share: pd.DataFrame) -> pd.DataFrame:
    fields = share.index.tolist()
    values = share.values
    dist = [[_jensen_shannon(values[i], values[j]) for j in range(len(fields))] for i in range(len(fields))]
    return pd.DataFrame(dist, index=fields, columns=fields)


def chart10_field_distance_heatmap(df: pd.DataFrame) -> None:
    """10. 분야 x 분야 거리 히트맵 - 11개 코드 비율 벡터 사이 Jensen-Shannon 거리. 분야 순서는
    기준 분야(사회학)와 가까운 순이라, 첫 행/열을 따라가면 사회학 대비 거리가 바로 읽힌다.
    진할수록 멀다(코드 구성이 다르다)."""
    share, n_codes = _field_code_share(df)
    dist = _field_distance_matrix(share)
    order = dist[REFERENCE_FIELD].sort_values(kind="stable").index.tolist()
    dist = dist.loc[order, order]

    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(dist.values, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
    labels = [f"{_FIELD_LABEL_WRAP.get(f, f)}\n(n={n_codes[f]})" for f in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{f}(n={n_codes[f]})" for f in order], fontsize=9)
    for i in range(len(order)):
        for j in range(len(order)):
            val = dist.values[i, j]
            if i == j:
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white" if val > 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Jensen-Shannon 거리 (0=같음, 1=완전히 다름)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "10_분야간_거리_히트맵.png", dpi=300)
    plt.close(fig)


def _classical_mds(dist: np.ndarray, dims: int = 2) -> np.ndarray:
    """거리 행렬을 평면 좌표로 옮기는 고전적 MDS(Torgerson). sklearn이 없어 numpy로 직접 계산한다."""
    n = dist.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    # numpy 2.2 + macOS Accelerate에서 작은 행렬곱에 divide by zero/overflow 가짜 경고가 뜬다 -
    # 결과는 정상(손 계산과 일치 확인)이라 이 계산에서만 경고를 끈다
    with np.errstate(all="ignore"):
        b = -0.5 * centering @ (dist**2) @ centering
    eigvals, eigvecs = np.linalg.eigh(b)
    top = np.argsort(eigvals)[::-1][:dims]
    return eigvecs[:, top] * np.sqrt(np.maximum(eigvals[top], 0))


def chart11_field_distance_map(df: pd.DataFrame) -> None:
    """11. 분야 거리 지도 - 10번 거리 행렬을 MDS로 평면에 옮긴 그림. 점 사이 거리가 코드 구성의
    차이를 나타내고, 점 크기는 코드 수(n)다. 기준 분야(사회학)는 주황으로 강조한다. MDS 좌표축
    자체에는 의미가 없어서(회전해도 같은 그림) 눈금을 뺀다. 원래 거리를 평면에 얼마나 담았는지
    (설명 비율)를 제목 아래에 적는다 - 이 비율이 낮으면 점 사이 거리를 곧이곧대로 읽으면 안 된다."""
    share, n_codes = _field_code_share(df)
    dist = _field_distance_matrix(share)
    coords = _classical_mds(dist.values)

    n = dist.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    with np.errstate(all="ignore"):  # _classical_mds와 같은 가짜 경고 억제
        b = -0.5 * centering @ (dist.values**2) @ centering
    eigvals = np.sort(np.linalg.eigvalsh(b))[::-1]
    explained = eigvals[:2].sum() / eigvals[eigvals > 0].sum() * 100

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sizes = {field: 40 + n_codes[field] * 12 for field in dist.index}
    x_span = coords[:, 0].max() - coords[:, 0].min()
    y_span = coords[:, 1].max() - coords[:, 1].min()
    for (x, y), field in zip(coords, dist.index):
        is_ref = field == REFERENCE_FIELD
        # 기준 분야는 다른 큰 점에 가려지지 않게 맨 위(zorder)에, 테두리를 진하게 그린다
        ax.scatter(
            x, y, s=sizes[field], color=ACCENT_COLOR if is_ref else BAR_COLOR,
            alpha=0.95 if is_ref else 0.8, edgecolor="#333333" if is_ref else "white",
            linewidth=1.2 if is_ref else 1.5, zorder=5 if is_ref else 3,
        )

    # 라벨 위치 규칙:
    # - 기준 분야(사회학)는 점 바로 위 - 기준점이라 다른 분야와 가까이 붙는 경우가 많아서
    # - 오른쪽 가까이에 다른 점이 있으면 라벨을 왼쪽에 둔다(오른쪽 점과 겹치지 않게)
    # - 나머지는 오른쪽. 오른쪽 라벨끼리 세로로 너무 붙으면 아래부터 훑으며 위로 밀어내고 선으로 잇는다
    def _crowded_on_right(i: int) -> bool:
        x, y = coords[i]
        return any(
            j != i and 0 < coords[j, 0] - x < x_span * 0.3 and abs(coords[j, 1] - y) < y_span * 0.08
            for j in range(len(coords))
        )

    side = {}
    for i, field in enumerate(dist.index):
        side[field] = "top" if field == REFERENCE_FIELD else ("left" if _crowded_on_right(i) else "right")

    min_gap = y_span * 0.07
    label_y = {}
    placed: list[tuple[float, float]] = []
    for idx in np.argsort(coords[:, 1]):
        field = dist.index[idx]
        if side[field] != "right":
            continue
        x, y = coords[idx]
        target = y
        for px, py in placed:
            if abs(px - x) < x_span * 0.35 and abs(py - target) < min_gap:
                target = py + min_gap
        placed.append((x, target))
        label_y[field] = target

    for (x, y), field in zip(coords, dist.index):
        is_ref = field == REFERENCE_FIELD
        label = f"{field}(n={n_codes[field]})"
        radius_pt = np.sqrt(sizes[field] / np.pi)
        style = {"fontsize": 9, "fontweight": "bold" if is_ref else "normal", "zorder": 6}
        if side[field] == "top":
            ax.annotate(label, (x, y), xytext=(0, radius_pt + 4), textcoords="offset points",
                        ha="center", va="bottom", **style)
        elif side[field] == "left":
            ax.annotate(label, (x, y), xytext=(-(radius_pt + 6), 0), textcoords="offset points",
                        ha="right", va="center", **style)
        elif abs(label_y[field] - y) > 1e-9:
            # 위로 밀어낸 라벨 - 옮긴 위치에 적고 점과 가는 선으로 잇는다
            ax.annotate(label, (x, y), xytext=(x + x_span * 0.06, label_y[field]), textcoords="data",
                        ha="left", va="center",
                        arrowprops={"arrowstyle": "-", "color": "#999999", "linewidth": 0.7}, **style)
        else:
            ax.annotate(label, (x, y), xytext=(radius_pt + 6, 0), textcoords="offset points",
                        ha="left", va="center", **style)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_color("#cccccc")
    ax.margins(x=0.08, y=0.12)
    # 라벨이 테두리 밖으로 나가지 않게 여백 추가 - 오른쪽 라벨은 항상, 왼쪽 라벨은 있을 때만
    left_pad = x_span * 0.35 if "left" in side.values() else 0
    ax.set_xlim(ax.get_xlim()[0] - left_pad, ax.get_xlim()[1] + x_span * 0.35)
    ax.set_title(f"평면이 원래 거리를 설명하는 비율: {explained:.0f}%", fontsize=9, color="#666666", loc="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "11_분야_거리지도_MDS.png", dpi=300)
    plt.close(fig)


def main() -> None:
    global RESULTS_CSV, OUT_DIR
    parser = argparse.ArgumentParser(description="분류 결과 CSV로 그래프를 그린다")
    parser.add_argument("--results", default=str(RESULTS_CSV), help="분류 결과 CSV (기본: 1회 분류 결과)")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="그래프 저장 폴더 (기본: charts/)")
    args = parser.parse_args()
    # 차트 함수들이 모듈 전역 RESULTS_CSV/OUT_DIR을 읽으므로 여기서 바꿔 준다 - 3회 결과는
    # --results classification_results_runs3.csv --out-dir charts_runs3 처럼 따로 그려서
    # 1회 결과 그래프와 나란히 비교할 수 있게 한다
    RESULTS_CSV, OUT_DIR = Path(args.results), Path(args.out_dir)
    print(f"입력: {RESULTS_CSV}")

    _setup_korean_font()
    OUT_DIR.mkdir(exist_ok=True)

    df = load_merged()
    print(f"병합된 논문 수(발행년 있는 것만): {len(df)}")

    chart0_classification_summary(df)
    chart1_overall_decade(df)
    chart2_field_decade_stacked(df)
    chart3_field_role_env_ratio_bar(df)
    chart3b_field_roleenv_diverging(df)
    chart4_field_all_codes_heatmap(df)
    chart4b_field_code_group_heatmap(df)
    chart5_field_fieldwindow_ratio_bar(df)
    chart3_5_combined(df)
    chart3_5_compact(df)
    chart5b_field_fieldwindow_diverging(df)
    chart_single_field_decade(df, "사회학", "6_사회학_시기별_추이.png")
    chart7_field_err_rate_bar(df)
    chart7_field_err_rate_bar(load_merged(excluded=True), "7c_화이트리스트_제외분야_ERR_비율.png")
    chart8_window_decade(df)
    chart12_golden_sociology_period()
    chart9_all_codes_count_bar(df)
    chart10_field_distance_heatmap(df)
    chart11_field_distance_map(df)

    print(f"완료 - {OUT_DIR} 아래 18개 PNG(+합본·압축본 PDF) 생성됨")


if __name__ == "__main__":
    main()
