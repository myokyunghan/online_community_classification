"""
ICR/ 폴더의 새 프롬프트(prompt_template_stage1.txt, prompt_template_stage2.txt)와
새 golden 데이터셋(온라인_커뮤니티_연구지형_분류논문_30편.csv)을 사용하는 2단계 분류 스크립트.

루트의 classify_paper_two_stage.py와 흐름은 동일하다 (1차 호출: 논문 원문 → focus,
2차 호출: focus만으로 11개 코드 판정 — 원본 초록은 2차 호출에 주지 않는다). 다만 이
폴더의 stage1 프롬프트는 focus를 "주제"/"결론"/"온라인 커뮤니티" 세 키를 가진 JSON
객체로 뽑고, stage2 프롬프트는 {{FOCUS}}에 그 객체를 JSON 그대로 삽입받는다는 점이
다르다 (루트 버전은 focus가 평문 한 문장이었다).

CSV의 "focus" 컬럼은 참고용 골드 데이터이며, 이 스크립트는 사용하지 않고 매번 1차
호출로 새로 생성한다 (루트 버전과 동일한 동작 — 모델이 실제로 무엇을 뽑아내는지를
보려는 것이므로 골드 focus로 대체하지 않는다).

사전 준비:
    export NVIDIA_API_KEY="nvapi-..."

사용 예:
    python3 ICR/classify_paper_two_stage.py --id 1
    python3 ICR/classify_paper_two_stage.py --paper-file paper.txt
    python3 ICR/classify_paper_two_stage.py --id 1 --reasoning-effort medium
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

# 루트에도 동명의 classify_paper_two_stage.py가 있으므로, 이 폴더(sys.path[0])가
# 먼저 검색되도록 루트 경로는 뒤에 append한다 (insert(0, ...)로 앞에 넣으면 이 파일
# 자신이 아니라 루트 버전이 잘못 로드될 위험이 있다).
sys.path.append(str(Path(__file__).resolve().parent.parent))

from classify_paper import MODEL, NVIDIA_BASE_URL, extract_json  # noqa: E402
from compare_with_golden import build_paper_text  # noqa: E402

ICR_DIR = Path(__file__).parent
TEMPLATE_STAGE1_PATH = ICR_DIR / "prompt_template_stage1.txt"
TEMPLATE_STAGE2_PATH = ICR_DIR / "prompt_template_stage2.txt"
DEFAULT_CSV = ICR_DIR / "온라인_커뮤니티_연구지형_분류논문_30편.csv"

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _response_candidates(text: str) -> list[str]:
    """모델이 ```json ... ``` 코드펜스로 감싸거나, 그 안 내용을 통째로 다시 이스케이프해서
    (실제 문자로 \\" \\n 등이 들어간) 내놓는 경우까지 파싱을 시도해볼 후보들을 만든다."""
    candidates = [text]
    match = CODE_FENCE_RE.search(text)
    fenced = match.group(1) if match else text
    if fenced != text:
        candidates.append(fenced)
    unescaped = fenced.replace('\\"', '"').replace("\\n", "\n")
    if unescaped not in candidates:
        candidates.append(unescaped)
    return candidates


def robust_json_loads(text: str) -> dict:
    last_exc = None
    for candidate in _response_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
    raise last_exc


def robust_extract_json(text: str) -> dict:
    last_exc = None
    for candidate in _response_candidates(text):
        try:
            return extract_json(candidate)
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
    raise last_exc


def build_stage1_messages(paper_text: str) -> list[dict]:
    """1차 호출: prompt_template_stage1.txt + 논문 원문 → focus 객체만 뽑는다."""
    template = TEMPLATE_STAGE1_PATH.read_text(encoding="utf-8")
    instructions, _, trailer = template.partition("{{PAPER_TEXT}}")
    return [
        {"role": "system", "content": instructions.rstrip()},
        {"role": "user", "content": paper_text + trailer},
    ]


FOCUS_BLOCK_MARKER = "[FOCUS]\n{{FOCUS}}"


def build_stage2_messages(focus: dict, fewshot_examples: list[tuple[dict, dict]] | None = None) -> list[dict]:
    """2차 호출: prompt_template_stage2.txt + focus 객체(원본 초록은 주지 않음) → STEP 1~3 수행.

    focus는 "주제"/"결론"/"온라인 커뮤니티" 키를 가진 dict — {{FOCUS}}엔 JSON 문자열로 삽입한다.

    fewshot_examples가 있으면 (데모용 focus, 이상적 출력) 쌍을 user/assistant 턴으로 실제
    질문 앞에 넣는다. 이 경우 system 메시지엔 특정 FOCUS를 미리 박아넣지 않는다 — 데모마다
    다른 FOCUS를 보여줘야 하므로, 실제 FOCUS는 대화의 마지막 user 턴에서만 준다."""
    template = TEMPLATE_STAGE2_PATH.read_text(encoding="utf-8")
    focus_json = json.dumps(focus, ensure_ascii=False, indent=2)

    if fewshot_examples:
        instructions = template.replace(
            FOCUS_BLOCK_MARKER, "[FOCUS]\n(아래 대화의 마지막 user 턴에서 실제 FOCUS가 제공된다)"
        )
    else:
        instructions = template.replace("{{FOCUS}}", focus_json)

    messages = [{"role": "system", "content": instructions}]
    for demo_focus, demo_output in fewshot_examples or []:
        messages.append({"role": "user", "content": json.dumps(demo_focus, ensure_ascii=False, indent=2)})
        messages.append({"role": "assistant", "content": json.dumps(demo_output, ensure_ascii=False)})
    messages.append({"role": "user", "content": focus_json})
    return messages


AUDIT_CODES = (
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
)


def parse_gold_focus_text(focus_text: str) -> dict:
    """golden CSV의 'focus' 컬럼("> 주제 : ...\\n> 결론 : ...\\n> 온라인 커뮤니티 : ...")을
    stage1 출력과 동일한 {"주제":..., "결론":..., "온라인 커뮤니티":...} dict로 변환한다.
    "결론" 줄이 비어 있는 행도 있으므로 없으면 빈 문자열로 둔다."""
    result = {"주제": "", "결론": "", "온라인 커뮤니티": ""}
    for line in (focus_text or "").splitlines():
        line = line.strip()
        if not line.startswith(">"):
            continue
        line = line[1:].strip()
        for key in result:
            prefix = f"{key} :"
            if line.startswith(prefix):
                result[key] = line[len(prefix):].strip()
                break
    return result


def _normalize_gold_code(code: str) -> str | None:
    """class1/class2가 'ROLE_'/'ENV_' 접두사 없이 들어온 경우까지 방어적으로 보정한다."""
    code = (code or "").strip().upper()
    if not code or code == "ERR":
        return None
    if code in AUDIT_CODES:
        return code
    for prefix in ("ROLE_", "ENV_"):
        if f"{prefix}{code}" in AUDIT_CODES:
            return f"{prefix}{code}"
    return None


def flatten_focus(focus: dict) -> str:
    """focus dict("주제"/"결론"/"온라인 커뮤니티")를 prompt_template_stage2.txt의 예시가 보여주는
    평문 문장으로 합친다. stage2 출력의 "focus" 필드는 항상 이 평문 형태여야 한다 — dict를 그대로
    넣으면(예시와 형식이 달라져서) 모델이 형식을 놓고 갈등하다 응답이 깨지는 현상이 확인됐다."""
    parts = [focus.get(k, "").strip() for k in ("주제", "결론", "온라인 커뮤니티")]
    return " ".join(p for p in parts if p)


def build_ideal_stage2_output(gold_row: dict) -> tuple[dict, dict]:
    """golden CSV 한 행(focus/class1/class2)으로부터 few-shot 데모용 (focus, 이상적 출력) 쌍을 만든다.
    반환하는 첫 번째 값(focus)은 실제 stage2 입력과 형식을 맞추기 위한 dict이고, 두 번째 값
    (이상적 출력)의 "focus" 필드는 위 flatten_focus()로 평문화한 문자열이다 — 프롬프트 예시와
    형식을 맞춰야 모델이 헷갈리지 않는다.

    quote는 규칙상 "focus의 주제 문장에서 나와야 한다"는 제약을 만족시키려고 그 논문의 주제
    문장을 그대로 쓴다 — 실제로 어느 부분이 근거인지 정교하게 고르는 건 사람 손이 필요하므로,
    이건 "이런 형식으로 답하라"는 최소한의 데모일 뿐, 근거 선정 자체의 모범답안은 아니다."""
    focus = parse_gold_focus_text(gold_row.get("focus", ""))
    granted = {
        c for c in (_normalize_gold_code(gold_row.get("class1", "")), _normalize_gold_code(gold_row.get("class2", "")))
        if c
    }

    topic = focus.get("주제", "")
    audit = []
    for code in AUDIT_CODES:
        if code in granted:
            audit.append({"code": code, "quote": topic, "prob": 0.85, "verdict": "부여"})
        else:
            audit.append({"code": code, "quote": None, "prob": 0.05, "verdict": "미부여"})

    if granted:
        rationale = f"[FOCUS]의 주제가 {', '.join(sorted(granted))}에 해당하는 근거를 담고 있으므로 해당 코드를 부여한다."
    else:
        rationale = "[FOCUS]가 11개 코드 중 어디에도 해당하는 구체적 근거를 담지 못해 전부 미부여로 판단한다."

    return focus, {"focus": flatten_focus(focus), "audit": audit, "rationale": rationale}


def _call(client: OpenAI, model: str, messages: list[dict], temperature: float, reasoning_effort: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=8192,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": reasoning_effort},
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        # gpt-oss(harmony 포맷)는 max_tokens를 다 쓰기 전에 reasoning만 채우고 최종 답을 못 내면
        # content=None을 준다 — finish_reason이 'length'면 토큰 부족이 원인이다.
        raise ValueError(
            f"API 응답 content가 비었습니다 (finish_reason={choice.finish_reason!r}, usage={response.usage})"
        )
    return content


def classify_paper_two_stage(
    paper_text: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    fewshot_examples: list[tuple[dict, dict]] | None = None,
) -> dict:
    """두 번의 API 호출로 논문을 분류한다. 반환값은 extract_json()을 거친 최종 dict에
    stage1_focus_raw(1차 호출 원문 응답)를 추가로 담는다.

    fewshot_examples는 build_ideal_stage2_output()으로 만든 (focus, 이상적 출력) 쌍의 리스트로,
    2차 호출에서만 사용한다 (1차 focus 추출은 여전히 zero-shot)."""
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    stage1_raw = _call(client, model, build_stage1_messages(paper_text), temperature, reasoning_effort)
    try:
        stage1_obj = robust_json_loads(stage1_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"1차 호출 응답이 유효한 JSON이 아닙니다 | 응답 일부: {stage1_raw[:500]!r}") from exc
    focus = stage1_obj.get("focus")
    if not focus:
        raise ValueError(f"1차 호출에서 focus를 얻지 못했습니다. 원문: {stage1_raw[:500]!r}")

    stage2_raw = _call(
        client, model, build_stage2_messages(focus, fewshot_examples), temperature, reasoning_effort
    )
    try:
        parsed = robust_extract_json(stage2_raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{exc} | 응답 일부: {stage2_raw[:500]!r}") from exc
    parsed["stage1_focus_raw"] = stage1_raw
    return parsed


def load_paper_text_by_id(csv_path: Path, paper_id: str) -> tuple[str, dict]:
    """ICR golden CSV에서 id로 논문을 찾아 build_paper_text() 형태로 변환한다.
    반환값은 (paper_text, gold_row) — gold_row에서 class1/class2/focus를 참고용으로 볼 수 있다."""
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["id"] == paper_id), None)
    if row is None:
        sys.exit(f"오류: id={paper_id} 논문을 golden CSV에서 찾을 수 없습니다.")
    return build_paper_text(row), row


def main():
    parser = argparse.ArgumentParser(description="ICR 프롬프트 기준 2단계(focus 먼저 → 그 focus만으로 코드 판정) 분류")
    parser.add_argument("--id", help="ICR golden CSV의 논문 id (--paper-file 대신 사용)")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="golden CSV 경로 (--id 사용 시)")
    parser.add_argument("--paper-file", help="논문 텍스트(제목/저자/초록) 파일 경로 (--id 대신 사용)")
    parser.add_argument("--api-key", default=None, help="NVIDIA API 키 (미지정 시 NVIDIA_API_KEY 환경변수 사용)")
    parser.add_argument("--output", default=None, help="결과 JSON을 저장할 파일 경로 (선택)")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["low", "medium", "high"],
        help="모델의 reasoning_effort (기본 low, 루트 classify_paper.py와 동일한 기본값)",
    )
    args = parser.parse_args()

    if not args.id and not args.paper_file:
        sys.exit("오류: --id 또는 --paper-file 중 하나는 필요합니다.")

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("오류: NVIDIA API 키가 없습니다. --api-key 또는 NVIDIA_API_KEY 환경변수를 설정하세요.")

    gold_row = None
    if args.id:
        paper_text, gold_row = load_paper_text_by_id(Path(args.csv), args.id)
        print(f"분류 중: id={args.id} - {gold_row['title'][:40]}...", file=sys.stderr)
    else:
        paper_text = Path(args.paper_file).read_text(encoding="utf-8")

    try:
        parsed = classify_paper_two_stage(
            paper_text, MODEL, api_key, reasoning_effort=args.reasoning_effort
        )
    except (ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"오류: {exc}")

    if gold_row:
        parsed = {
            "id": gold_row["id"],
            "title": gold_row["title"],
            "class1_raw": gold_row.get("class1", ""),
            "class2_raw": gold_row.get("class2", ""),
            "gold_focus": gold_row.get("focus", ""),
            "model_response": parsed,
        }

    pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
    print(pretty)

    if args.output:
        Path(args.output).write_text(pretty, encoding="utf-8")
        print(f"\n결과가 {args.output} 에 저장되었습니다.", file=sys.stderr)


if __name__ == "__main__":
    main()
