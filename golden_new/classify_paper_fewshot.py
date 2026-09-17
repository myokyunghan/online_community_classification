"""
3회 호출 구조로 논문을 분류한다.

    1차 호출 (prompt_template_division.txt): 논문 텍스트만 보고 (a) 온라인 커뮤니티 정의에
        해당하는 대상이 실제로 있는지, (b) 이 논문이 커뮤니티의 "역할"(창/장)을 조명하는지,
        (c) 커뮤니티의 "환경적 조건"(회원제/거버넌스/생애주기/익명성)을 조명하는지 판단한다.
        (b), (c)는 동시에 해당할 수 있다. 커뮤니티가 없거나 (b)(c) 둘 다 아니면 그대로 ERR.
    2차 호출 (prompt_template_role.txt, (b)가 true일 때만): 창/장 반사실 테스트를 적용하며
        ROLE 6개 코드만 채점한다.
    3차 호출 (prompt_template_env.txt, (c)가 true일 때만): ENV 4개 코드만 채점한다.

2·3차에서 나온 audit(각각 최대 6개, 4개)를 합쳐서 최종적으로 상위 2개(0.5 이상인 것만)를
뽑는다 — "합쳐서 최대 2개"라는 캡은 ROLE/ENV를 나눠 호출해도 전체 기준으로 그대로 유지된다.

10개를 한 번에 판단하게 하면 부담이 크다는 문제, 그리고 이전의 2단계(축 결정→코드판정)
캐스케이드에서 "1차가 틀리면 2차가 못 되돌리는" 문제(특히 WINDOW/FIELD 판단을 1차에서
확정해버려서 2차 코드 판정을 막아버린 것)를 이렇게 다시 나눠서 완화한다 — 창/장 반사실
테스트는 이제 ROLE 호출 안에서 직접 이뤄지므로, 그 판단이 코드 채점과 분리되지 않는다.

이전 실험(ICR/, gpt-oss-120b 기준)에서 배운 것 중 이어받은 것:
    - few-shot 데모와 실제 프롬프트의 출력 형식(키 구성)을 반드시 똑같이 맞춘다.
    - 이 모델은 코드펜스로 감싸거나 그 안을 다시 이스케이프해서 내놓는 경우가 있어,
      코드펜스 벗기기+이스케이프 복구를 시도하는 robust 파서를 쓴다.

ICR/과 달리 바꾼 것: few-shot을 시스템 프롬프트 안에 텍스트로 뭉쳐넣지 않고, 실제
user/assistant 대화 턴으로 넣는다(`_fewshot_turns`). gpt-oss-120b는 멀티턴을 쓰면
finish_reason='stop'인데 content가 비는 문제가 있어 텍스트 삽입 방식을 썼지만, 여기서는
그 문제가 없다는 전제로 일반적인 chat 형식 few-shot을 쓴다.

자체 서버(143.248.248.192, vLLM으로 띄운 mlx-community/Qwen3.8-27B-4bit)를 호출한다 — 인증이 필요 없어 API 키
설정은 불필요하다. NVIDIA_BASE_URL/MODEL은 이 폴더의 classify_paper.py에서 정의한다.

사용 예:
    python3 golden_new/classify_paper_fewshot.py --id 1
    python3 golden_new/classify_paper_fewshot.py --paper-file paper.txt
    python3 golden_new/classify_paper_fewshot.py --id 1 --reasoning-effort medium
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classify_paper import DUMMY_API_KEY, MODEL, NVIDIA_BASE_URL, derive_role_env_from_audit  # noqa: E402

GOLDEN_NEW_DIR = Path(__file__).parent


def build_abstract_only_text(row: dict) -> str:
    """제목/저자/출처 없이 초록만 모델에 넘긴다 — 제목에 "온라인 커뮤니티"가 언급되는 것만
    보고 실제 초록엔 없는데도 has_community를 true로 잘못 판단하는 현상(id=9)이 있어서,
    제목 등 부가정보 없이 초록 내용만으로 판단하게 한다."""
    return (row.get("abstract") or "").strip()


DIVISION_PROMPT_PATH = GOLDEN_NEW_DIR / "prompt_template_division.txt"
ROLE_PROMPT_PATH = GOLDEN_NEW_DIR / "prompt_template_role.txt"
ENV_PROMPT_PATH = GOLDEN_NEW_DIR / "prompt_template_env.txt"
DEFAULT_CSV = GOLDEN_NEW_DIR / "온라인_커뮤니티_연구지형_분류논문_30편.csv"

DIVISION_PROMPT = DIVISION_PROMPT_PATH.read_text(encoding="utf-8")
ROLE_PROMPT = ROLE_PROMPT_PATH.read_text(encoding="utf-8")
ENV_PROMPT = ENV_PROMPT_PATH.read_text(encoding="utf-8")

ROLE_CODES = (
    "ROLE_FIELD_PUBLICSPHERE",
    "ROLE_FIELD_SOCIALCAPITAL",
    "ROLE_WINDOW_POLARIZE",
    "ROLE_WINDOW_GENDER",
    "ROLE_WINDOW_HATE",
    "ROLE_WINDOW_MISC",
)
ENV_CODES = (
    "ENV_COMMUNITY_SUBSCRIPTION",
    "ENV_COMMUNITY_GOVERNANCE",
    "ENV_COMMUNITY_LIFECYCLE",
    "ENV_ONLINE_ANNONIMITY",
)
AUDIT_CODES = ROLE_CODES + ENV_CODES  # ENV_COMMUNITY_DEMOGRAPHIC은 의도적으로 제외한 10개

AUDIT_REQUIRED_KEYS = ("audit", "rationale")
DIVISION_REQUIRED_KEYS = ("has_community", "role_relevant", "env_relevant")

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


def parse_gold_focus_text(focus_text: str) -> dict:
    """golden CSV의 'focus' 컬럼("> 주제 : ...\\n> 결론 : ...\\n> 온라인 커뮤니티 : ...")을
    {"주제":..., "결론":..., "온라인 커뮤니티":...} dict로 변환한다."""
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


def extract_raw_audit(text: str) -> dict:
    """audit/rationale만 파싱한다 (derive_role_env_from_audit는 아직 적용하지 않음 —
    ROLE 호출과 ENV 호출의 audit를 합친 뒤 한 번에 적용해야 '합쳐서 최대 2개' 캡이
    전체 10개 기준으로 정확히 작동한다)."""
    start = text.find("{")
    if start == -1:
        raise ValueError("응답에서 JSON 객체를 찾을 수 없습니다.")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    missing = [key for key in AUDIT_REQUIRED_KEYS if key not in obj]
    if missing:
        raise ValueError(
            f"응답이 불완전합니다 (누락된 키: {missing}). 모델이 전체 출력 스키마를 완성하지 못하고 중간에 끊긴 것으로 보입니다."
        )
    if not isinstance(obj.get("audit"), list) or not obj["audit"]:
        raise ValueError("응답의 audit 배열이 비어있거나 형식이 올바르지 않습니다.")
    return obj


def robust_extract_raw_audit(text: str) -> dict:
    last_exc = None
    for candidate in _response_candidates(text):
        try:
            return extract_raw_audit(candidate)
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
    raise last_exc


def extract_division_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("1차 판단 응답에서 JSON 객체를 찾을 수 없습니다.")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    missing = [key for key in DIVISION_REQUIRED_KEYS if key not in obj]
    if missing:
        raise ValueError(f"1차 판단 응답이 불완전합니다 (누락된 키: {missing}).")
    return obj


def robust_extract_division_json(text: str) -> dict:
    last_exc = None
    for candidate in _response_candidates(text):
        try:
            return extract_division_json(candidate)
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
    raise last_exc


def _build_ideal_partial(gold_row: dict, codes: tuple[str, ...]) -> dict:
    """golden CSV 한 행으로부터 ROLE 또는 ENV 전용 few-shot 데모 출력을 만든다.
    quote/rationale은 division_reasoning 컬럼(사람이 어떤 코드가 왜 부여됐는지 직접 써둔 것)이
    있으면 그걸 그대로 쓴다 — division 단계 few-shot과 동일한 근거 문장을 재사용해서, 1차
    (role/env 여부 판단)와 2·3차(구체적 코드 선택) 단계가 서로 다른 근거로 따로 노는 일이
    없게 한다. 없으면 focus 컬럼의 "주제" 문장으로 대체한다(형식 데모일 뿐, 근거 선정 자체의
    모범답안은 아니다)."""
    focus = parse_gold_focus_text(gold_row.get("focus", ""))
    granted_all = {
        c
        for c in (_normalize_gold_code(gold_row.get("class1", "")), _normalize_gold_code(gold_row.get("class2", "")))
        if c
    }
    granted = granted_all & set(codes)

    topic = focus.get("주제", "")
    quote = (gold_row.get("division_reasoning") or "").strip() or topic

    audit = []
    for code in codes:
        if code in granted:
            audit.append({"code": code, "quote": quote, "prob": 0.85, "verdict": "부여"})
        else:
            audit.append({"code": code, "quote": None, "prob": 0.05, "verdict": "미부여"})

    if granted:
        rationale = quote
    else:
        rationale = "이 코드군 중 어디에도 해당하는 구체적 근거를 담지 못해 전부 미부여로 판단한다."

    return {"audit": audit, "rationale": rationale}


def build_ideal_role_output(gold_row: dict) -> dict:
    return _build_ideal_partial(gold_row, ROLE_CODES)


def build_ideal_env_output(gold_row: dict) -> dict:
    return _build_ideal_partial(gold_row, ENV_CODES)


def build_ideal_division_output(gold_row: dict) -> dict:
    """golden CSV 한 행으로부터 1차(division) 단계 few-shot 데모 출력을 만든다.
    class1이 ERR이면 커뮤니티가 없거나 역할/환경 둘 다 아닌 것이고, 아니면 class1/class2가
    ROLE_CODES/ENV_CODES 중 어디에 속하는지로 role_relevant/env_relevant를 그대로 도출한다.
    reasoning은 golden CSV의 division_reasoning 컬럼(사람이 창/장 여부를 직접 써둔 것)이
    있으면 그걸 그대로 쓰고, 없으면 focus 컬럼의 "주제" 문장으로 대체한다."""
    focus = parse_gold_focus_text(gold_row.get("focus", ""))
    topic = focus.get("주제", "")
    is_err = gold_row.get("class1", "").strip().upper() == "ERR"

    if is_err:
        return {
            "has_community": False,
            "role_relevant": False,
            "env_relevant": False,
            "reasoning": topic or "온라인 커뮤니티를 분석 대상으로 삼지 않거나, 역할/환경 어디에도 해당하지 않는다.",
        }

    granted_all = {
        c
        for c in (_normalize_gold_code(gold_row.get("class1", "")), _normalize_gold_code(gold_row.get("class2", "")))
        if c
    }
    reasoning = (gold_row.get("division_reasoning") or "").strip() or topic or "논문 내용이 역할 또는 환경 판단 기준에 해당하는 근거를 담고 있다."

    return {
        "has_community": True,
        "role_relevant": bool(granted_all & set(ROLE_CODES)),
        "env_relevant": bool(granted_all & set(ENV_CODES)),
        "reasoning": reasoning,
    }


def _fewshot_turns(fewshot_examples: list[tuple[str, dict]]) -> list[dict]:
    """few-shot 예시를 실제 user/assistant 대화 턴으로 만든다 (ICR/와 달리 시스템 프롬프트에
    텍스트로 뭉쳐넣지 않음). ICR/에서는 gpt-oss-120b가 멀티턴을 쓰면 finish_reason='stop'인데
    content가 비는 현상이 있어 텍스트 삽입 방식을 썼지만, golden_new는 다른 모델(Qwen, vLLM
    경유)이라 그 문제가 없다는 전제로 일반적인 few-shot 대화 형식을 쓴다."""
    turns = []
    for demo_text, demo_output in fewshot_examples:
        turns.append({"role": "user", "content": demo_text})
        turns.append({"role": "assistant", "content": json.dumps(demo_output, ensure_ascii=False)})
    return turns


def build_division_messages(paper_text: str, fewshot_examples: list[tuple[str, dict]] | None = None) -> list[dict]:
    return [
        {"role": "system", "content": DIVISION_PROMPT},
        *(_fewshot_turns(fewshot_examples) if fewshot_examples else []),
        {"role": "user", "content": paper_text},
    ]


def build_role_messages(paper_text: str, fewshot_examples: list[tuple[str, dict]] | None = None) -> list[dict]:
    return [
        {"role": "system", "content": ROLE_PROMPT},
        *(_fewshot_turns(fewshot_examples) if fewshot_examples else []),
        {"role": "user", "content": paper_text},
    ]


def build_env_messages(paper_text: str, fewshot_examples: list[tuple[str, dict]] | None = None) -> list[dict]:
    return [
        {"role": "system", "content": ENV_PROMPT},
        *(_fewshot_turns(fewshot_examples) if fewshot_examples else []),
        {"role": "user", "content": paper_text},
    ]


def build_err_result(reasoning: str) -> dict:
    """1차 판단에서 커뮤니티가 없거나 역할/환경 둘 다 아니라고 나오면, 2·3차 호출 없이 바로
    ERR로 끝낸다."""
    audit = [{"code": code, "quote": None, "prob": 0.02, "verdict": "미부여"} for code in AUDIT_CODES]
    obj = {
        "audit": audit,
        "rationale": reasoning or "1차 판단에서 온라인 커뮤니티가 없거나 역할/환경 어디에도 해당하지 않는다고 판단됨.",
    }
    return derive_role_env_from_audit(obj)


def _call(client: OpenAI, model: str, messages: list[dict], temperature: float, reasoning_effort: str) -> str:
    """reasoning_effort는 인자로 받되 실제로는 쓰지 않는다 — NVIDIA gpt-oss(harmony 포맷)
    전용 extra_body 파라미터였고, vLLM으로 띄운 Qwen 서버는 이걸 모르므로 요청에서 뺐다.
    호출부(CLI --reasoning-effort 등)와의 시그니처 호환을 위해 인자 자체는 남겨둔다."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        raise ValueError(
            f"API 응답 content가 비었습니다 (finish_reason={choice.finish_reason!r}, usage={response.usage})"
        )
    return content


def classify_paper_fewshot(
    paper_text: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    fewshot_examples: list[tuple[str, dict]] | None = None,
    env_fewshot_examples: list[tuple[str, dict]] | None = None,
    division_fewshot_examples: list[tuple[str, dict]] | None = None,
) -> dict:
    """1차(분리 판단) → 2차(ROLE, 조건부) → 3차(ENV, 조건부) 순으로 논문을 분류한다.
    division_fewshot_examples는 1차(분리) 호출용, fewshot_examples는 ROLE 호출용,
    env_fewshot_examples는 ENV 호출용 — 다 (논문 텍스트, 이상적 출력) 쌍 리스트다."""
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    division_raw = _call(
        client,
        model,
        build_division_messages(paper_text, division_fewshot_examples),
        temperature,
        reasoning_effort,
    )
    try:
        division = robust_extract_division_json(division_raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"1차(분리) 호출 파싱 실패: {exc} | 응답 일부: {division_raw[:500]!r}") from exc

    role_relevant = bool(division.get("role_relevant"))
    env_relevant = bool(division.get("env_relevant"))

    if not division.get("has_community") or not (role_relevant or env_relevant):
        result = build_err_result(division.get("reasoning", ""))
        result["division"] = division
        return result

    combined_audit = []
    rationales = []

    if role_relevant:
        role_raw = _call(client, model, build_role_messages(paper_text, fewshot_examples), temperature, reasoning_effort)
        try:
            role_obj = robust_extract_raw_audit(role_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"2차(ROLE) 호출 파싱 실패: {exc} | 응답 일부: {role_raw[:500]!r}") from exc
        combined_audit.extend(item for item in role_obj["audit"] if item.get("code") in ROLE_CODES)
        rationales.append(f"[ROLE] {role_obj.get('rationale', '')}")

    if env_relevant:
        env_raw = _call(client, model, build_env_messages(paper_text, env_fewshot_examples), temperature, reasoning_effort)
        try:
            env_obj = robust_extract_raw_audit(env_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"3차(ENV) 호출 파싱 실패: {exc} | 응답 일부: {env_raw[:500]!r}") from exc
        combined_audit.extend(item for item in env_obj["audit"] if item.get("code") in ENV_CODES)
        rationales.append(f"[ENV] {env_obj.get('rationale', '')}")

    parsed = derive_role_env_from_audit({"audit": combined_audit, "rationale": " | ".join(rationales)})
    parsed["division"] = division
    return parsed


def load_paper_text_by_id(csv_path: Path, paper_id: str) -> tuple[str, dict]:
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["id"] == paper_id), None)
    if row is None:
        sys.exit(f"오류: id={paper_id} 논문을 golden CSV에서 찾을 수 없습니다.")
    return build_abstract_only_text(row), row


def main():
    parser = argparse.ArgumentParser(description="1차(분리)+2차(ROLE)+3차(ENV) 구조로 논문 분류")
    parser.add_argument("--id", help="ICR golden CSV의 논문 id (--paper-file 대신 사용)")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="golden CSV 경로 (--id 사용 시)")
    parser.add_argument("--paper-file", help="논문 텍스트(제목/저자/초록) 파일 경로 (--id 대신 사용)")
    parser.add_argument(
        "--api-key", default=None, help="자체 서버 API 키 (인증 불필요 — 미지정 시 더미 값 사용)"
    )
    parser.add_argument("--output", default=None, help="결과 JSON을 저장할 파일 경로 (선택)")
    parser.add_argument(
        "--reasoning-effort", default="low", choices=["low", "medium", "high"], help="모델의 reasoning_effort (기본 low, 현재 서버엔 미적용)"
    )
    args = parser.parse_args()

    if not args.id and not args.paper_file:
        sys.exit("오류: --id 또는 --paper-file 중 하나는 필요합니다.")

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY") or DUMMY_API_KEY

    gold_row = None
    if args.id:
        paper_text, gold_row = load_paper_text_by_id(Path(args.csv), args.id)
        print(f"분류 중: id={args.id} - {gold_row['title'][:40]}...", file=sys.stderr)
    else:
        paper_text = Path(args.paper_file).read_text(encoding="utf-8")

    try:
        parsed = classify_paper_fewshot(paper_text, MODEL, api_key, reasoning_effort=args.reasoning_effort)
    except (ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"오류: {exc}")

    if gold_row:
        parsed = {
            "id": gold_row["id"],
            "title": gold_row["title"],
            "class1_raw": gold_row.get("class1", ""),
            "class2_raw": gold_row.get("class2", ""),
            "model_response": parsed,
        }

    pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
    print(pretty)

    if args.output:
        Path(args.output).write_text(pretty, encoding="utf-8")
        print(f"\n결과가 {args.output} 에 저장되었습니다.", file=sys.stderr)


if __name__ == "__main__":
    main()
