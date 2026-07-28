"""
golden CSV에서 특정 id의 논문 하나만 골라 NVIDIA API로 분류하고,
모델의 전체 JSON 응답(주분류/보조분류 이유와 근거 포함)을 파일로 저장한다.
comparison_results.csv에서 특정 논문의 예측이 정답과 왜 어긋났는지 확인할 때 사용한다.

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."

사용 예:
    python inspect_paper.py --id 2
    python inspect_paper.py --id 2 --output golden/inspect_2.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from compare_with_golden import DEFAULT_CSV, classify_row_with_retry


def main():
    parser = argparse.ArgumentParser(description="golden CSV의 논문 하나를 NVIDIA API로 분류하고 전체 JSON을 저장")
    parser.add_argument("--id", required=True, help="golden CSV의 논문 id")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="정답 CSV 경로")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--output", default=None, help="결과 JSON을 저장할 파일 경로 (기본: golden/inspect_<id>.json)")
    parser.add_argument("--max-retries", type=int, default=3, help="일시적 오류/파싱 실패 시 최대 재시도 횟수 (기본 3)")
    parser.add_argument("--retry-wait-minutes", type=float, default=3.0, help="서버 오류 재시도 전 대기 시간(분, 기본 3.0)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    row = next((r for r in rows if r["id"] == args.id), None)
    if row is None:
        sys.exit(f"오류: id={args.id} 논문을 찾을 수 없습니다.")

    print(f"분류 중: id={row['id']} - {row['title']}", file=sys.stderr)
    prediction = classify_row_with_retry(row, api_key, args.max_retries, args.retry_wait_minutes * 60)

    result = {
        "id": row["id"],
        "title": row["title"],
        "class1_raw": row.get("class1", ""),
        "class2_raw": row.get("class2", ""),
        "model_response": prediction["parsed"],
    }
    pretty = json.dumps(result, ensure_ascii=False, indent=2)

    output_path = Path(args.output) if args.output else Path("golden") / f"inspect_{args.id}.json"
    output_path.write_text(pretty, encoding="utf-8")

    print(pretty)
    print(f"\n결과 저장: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
