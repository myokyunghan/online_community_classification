"""
golden 디렉토리의 CSV(22편 논문, 제목+초록+class1/class2 정답)를 읽어
NVIDIA NIM API로 각 논문을 분류시킨 뒤, 예측 결과를 정답과 코드 단위로 비교한다.

prompt_template.txt는 별도의 사전 필터링 없이 STEP 1(초점 요약) → 2(ROLE 채점) →
3(ENV 채점) → 4(출력) 순서로 판정하고, focus/audit/rationale을 반환한다. ROLE 축
(담론장/사회자본/양극화/젠더/혐오/기타창) 6개와 ENV 축(구독제/거버넌스/인구통계/
생애주기/익명성) 5개, 총 11개 코드를 audit에서 전수 평가하며, "부여"로 판정한
코드에는 그 자리에서 확률(prob)도 함께 매긴다.

모델이 role/env를 따로 조립하지는 않는다 — classify_paper.py의
derive_role_env_from_audit()이 audit에서 "부여"로 판정된 코드와 prob을 직접 모아
role/env를 만든다 (확률 상위 2개만, 축은 코드 접두사 ROLE_/ENV_로 구분). 이렇게
하면 모델이 audit 판정과 다르게 role/env를 조립하다 생기는 누락·불일치가 애초에
생기지 않는다. 부여된 코드가 3개 이상이면 확률 하위 코드는 잘려나가며,
_dropped_by_cap에 기록되어 compare_row()의 dropped_by_cap_runs/gold_dropped_by_cap
으로 추적된다 — 정답 코드가 캡 때문에 잘렸는지 바로 확인할 수 있다. 11개 코드가
전부 미부여로 끝나면 status가 자연스럽게 "ERR"이 된다(별도 게이트 없음).

CSV의 class1/class2는 이제 새 11개 코드(ROLE_FIELD_PUBLICSPHERE, ENV_ONLINE_ANNONIMITY
등)를 직접 담고 있으므로 별도 매핑 없이 그대로 사용한다.

class1이 "ERR"인 행은 온라인 커뮤니티 연구가 아닌 논문(분류 불가 테스트 케이스)이다.
이 경우 정답 비교는 코드 집합이 아니라 모델이 반환한 status가 다수결로 "ERR"인지를
기준으로 한다.

class1/class2는 우선순위 없는 '해당 논문에 적용되는 카테고리 집합'이므로,
정답과 예측을 순서 없이 집합으로 비교한다.

temperature=0으로도 모델 응답이 매번 완전히 동일하지는 않을 수 있으나, 기본값은
속도를 위해 논문당 1회만 호출한다. 다수결 안정성 검증이 필요하면
--runs-per-paper로 늘릴 수 있다.

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."
    pip install -r requirements.txt

사용 예:
    python compare_with_golden.py
    python compare_with_golden.py --limit 3          # 앞 3편만 테스트
    python compare_with_golden.py --sample 5         # 무작위 5편만 테스트
    python compare_with_golden.py --output result.csv
    python compare_with_golden.py --runs-per-paper 5
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError

from classify_paper import MODEL, classify_paper, extract_json

DEFAULT_CSV = Path(__file__).parent / "golden" / "온라인_커뮤니티_연구지형_분류논문_22편.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "golden" / "comparison_results.csv"

RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, TimeoutError, RateLimitError, InternalServerError)
PARSE_RETRY_WAIT_SECONDS = 2.0


def classify_retry_kind(exc: Exception) -> str | None:
    """재시도 종류를 판정한다: 'server'(긴 대기 후 재시도), 'parse'(짧은 대기 후 재시도), None(재시도 안 함)."""
    if isinstance(exc, RETRYABLE_ERRORS):
        return "server"
    # NVIDIA NIM은 모델 배포가 일시적으로 불안정할 때 400 + "DEGRADED"를 반환한다.
    # 상태 코드는 400(Bad Request)이지만 실제로는 일시적 서버 문제이므로 재시도한다.
    if isinstance(exc, BadRequestError) and "DEGRADED" in str(exc):
        return "server"
    # 모델이 원문의 따옴표를 이스케이프하지 않는 등, 생성한 JSON 자체가 깨진 경우.
    # 서버 문제가 아니라 그 순간의 생성 결과 문제이므로 오래 기다릴 필요 없이 바로 재시도한다.
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "parse"
    return None


# prompt_template.txt의 분류 체계 표와 동일해야 한다
ROLE_CODES = {
    "ROLE_FIELD_PUBLICSPHERE",
    "ROLE_FIELD_SOCIALCAPITAL",
    "ROLE_WINDOW_POLARIZE",
    "ROLE_WINDOW_GENDER",
    "ROLE_WINDOW_HATE",
    "ROLE_WINDOW_MISC",
}
ENV_CODES = {
    "ENV_COMMUNITY_SUBSCRIPTION",
    "ENV_COMMUNITY_GOVERNANCE",
    "ENV_COMMUNITY_DEMOGRAPHIC",
    "ENV_COMMUNITY_LIFECYCLE",
    "ENV_ONLINE_ANNONIMITY",
}
VALID_CODES = ROLE_CODES | ENV_CODES

def label_to_code(label: str) -> str | None:
    """golden CSV의 class1/class2는 이제 코드를 직접 담고 있으므로 유효성만 검증한다."""
    label = (label or "").strip()
    return label if label in VALID_CODES else None


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_paper_text(row: dict) -> str:
    return (
        f"제목: {row['title']}\n"
        f"저자: {row['author']} ({row['published_year']})\n"
        f"출처: {row['journal']}\n\n"
        f"초록:\n{row['abstract']}"
    )


def extract_axis_codes(parsed: dict, axis: str, valid: set) -> list[str]:
    items = parsed.get(axis, [])
    return [item.get("code", "").strip() for item in items if item.get("code", "").strip() in valid]


def classify_row(row: dict, api_key: str, temperature: float = 0.0) -> dict:
    raw_output = classify_paper(build_paper_text(row), MODEL, api_key, temperature=temperature)
    try:
        parsed = extract_json(raw_output)
    except (ValueError, json.JSONDecodeError) as exc:
        snippet = raw_output[:500] if raw_output else "(빈 응답)"
        raise ValueError(f"{exc} | 응답 일부: {snippet!r}") from exc

    role_codes = extract_axis_codes(parsed, "role", ROLE_CODES)
    env_codes = extract_axis_codes(parsed, "env", ENV_CODES)
    predicted_codes = set(role_codes) | set(env_codes)
    dropped_by_cap = set(parsed.get("_dropped_by_cap") or [])

    return {
        "parsed": parsed,
        "status": parsed.get("status", "OK"),
        "focus": parsed.get("focus"),
        "role_codes": role_codes,
        "env_codes": env_codes,
        "predicted_codes": predicted_codes,
        "dropped_by_cap": dropped_by_cap,
    }


def classify_row_with_retry(row: dict, api_key: str, max_retries: int, retry_wait_seconds: float) -> dict:
    """서버 오류는 긴 대기 후, JSON 파싱 실패는 짧은 대기 후 재시도한다."""
    for attempt in range(1, max_retries + 1):
        try:
            return classify_row(row, api_key)
        except Exception as exc:
            kind = classify_retry_kind(exc)
            if kind is None or attempt >= max_retries:
                raise
            if kind == "server":
                wait, label = retry_wait_seconds, f"{retry_wait_seconds / 60:.1f}분"
            else:
                wait, label = PARSE_RETRY_WAIT_SECONDS, f"{PARSE_RETRY_WAIT_SECONDS:.0f}초"
            print(
                f"    일시적 오류 (시도 {attempt}/{max_retries}) - {label} 후 재시도: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)


def classify_row_multi(
    row: dict, api_key: str, runs: int, max_retries: int, retry_wait_seconds: float, sleep_seconds: float
) -> list[dict]:
    predictions = []
    for run in range(1, runs + 1):
        predictions.append(classify_row_with_retry(row, api_key, max_retries, retry_wait_seconds))
        if run < runs:
            time.sleep(sleep_seconds)
    return predictions


def aggregate_predictions(predictions: list[dict]) -> dict:
    """3회 호출 결과 중 과반수(run 수의 절반 초과)에서 나온 코드만 최종 예측으로 채택한다."""
    run_code_sets = [p["predicted_codes"] for p in predictions]

    counts = {}
    for codes in run_code_sets:
        for code in codes:
            counts[code] = counts.get(code, 0) + 1

    n = len(predictions)
    majority_codes = {code for code, cnt in counts.items() if cnt * 2 > n}
    full_agreement = len({frozenset(codes) for codes in run_code_sets}) == 1

    err_count = sum(1 for p in predictions if p["status"] == "ERR")
    majority_status = "ERR" if err_count * 2 > n else "OK"

    return {
        "run_code_sets": run_code_sets,
        "majority_codes": majority_codes,
        "full_agreement": full_agreement,
        "majority_status": majority_status,
        "err_count": err_count,
    }


def compare_row(row: dict, predictions: list[dict], aggregate: dict) -> dict:
    """class1/class2는 우선순위 없는 '해당하는 카테고리 집합'이므로 집합 단위로 비교한다.
    class1이 "ERR"이면 코드 집합이 아니라 모델의 status 다수결이 "ERR"인지로 판정한다."""
    class1_raw = row.get("class1", "").strip()
    class2_raw = row.get("class2", "").strip()
    is_err_gold = class1_raw.upper() == "ERR"

    predicted_codes = aggregate["majority_codes"]
    run_details = "; ".join(
        f"run{i}:{','.join(sorted(codes)) if codes else '없음'}"
        for i, codes in enumerate(aggregate["run_code_sets"], 1)
    )
    rationales = " | ".join(str(p["parsed"].get("rationale", "")) for p in predictions)
    focuses = " | ".join(str(p.get("focus") or "") for p in predictions)
    error_runs = f"{aggregate['err_count']}/{len(predictions)}"

    dropped_runs = [p.get("dropped_by_cap") or set() for p in predictions]
    dropped_by_cap_runs = "; ".join(
        f"run{i}:{','.join(sorted(d)) if d else '없음'}" for i, d in enumerate(dropped_runs, 1)
    )

    if is_err_gold:
        exact_match = aggregate["majority_status"] == "ERR"
        gold_covered = exact_match
        class1_code, class2_code, gold_codes_display = "ERR", "", "ERR"
        gold_codes = set()
    else:
        class1_code = label_to_code(class1_raw)
        class2_code = label_to_code(class2_raw) if class2_raw else None
        gold_codes = {c for c in (class1_code, class2_code) if c}
        exact_match = bool(gold_codes) and gold_codes == predicted_codes
        gold_covered = bool(gold_codes) and gold_codes.issubset(predicted_codes)
        gold_codes_display = ", ".join(sorted(gold_codes)) if gold_codes else ""
        class1_code = class1_code or "미매핑"
        class2_code = class2_code or ""

    # 캡(최대 2개)이 잘라낸 코드 중에 정답이 있었는지 — 있었다면 캡 때문에 정답을 놓친 것이다.
    gold_dropped_by_cap = any(gold_codes & d for d in dropped_runs)

    return {
        "id": row["id"],
        "title": row["title"],
        "class1_raw": class1_raw,
        "class1_code": class1_code,
        "class2_raw": class2_raw,
        "class2_code": class2_code,
        "gold_codes": gold_codes_display,
        "majority_status": aggregate["majority_status"],
        "majority_codes": ", ".join(sorted(predicted_codes)) if predicted_codes else "",
        "run_details": run_details,
        "full_agreement": aggregate["full_agreement"],
        "error_runs": error_runs,
        "dropped_by_cap_runs": dropped_by_cap_runs,
        "gold_dropped_by_cap": gold_dropped_by_cap,
        "exact_match": exact_match,
        "gold_covered": gold_covered,
        "focuses": focuses,
        "rationales": rationales,
    }


def main():
    parser = argparse.ArgumentParser(description="golden CSV 기반 NVIDIA API 분류 정확도 비교")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="정답 CSV 경로")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="비교 결과를 저장할 CSV 경로")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--limit", type=int, default=None, help="테스트용으로 앞 N편만 처리")
    parser.add_argument("--sample", type=int, default=None, help="전체 중 N편을 무작위로 뽑아 테스트")
    parser.add_argument("--seed", type=int, default=None, help="--sample 사용 시 재현 가능한 무작위 시드")
    parser.add_argument("--runs-per-paper", type=int, default=1, help="논문당 반복 호출 횟수, 과반수로 채택 (기본 1)")
    parser.add_argument("--sleep", type=float, default=1.0, help="API 호출 사이 대기 시간(초, 기본 1.0)")
    parser.add_argument("--max-retries", type=int, default=3, help="타임아웃/연결 오류 시 최대 재시도 횟수 (기본 3)")
    parser.add_argument(
        "--retry-wait-minutes", type=float, default=3.0, help="재시도 전 대기 시간(분, 기본 3.0)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

    rows = load_rows(Path(args.csv))
    if args.sample:
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
