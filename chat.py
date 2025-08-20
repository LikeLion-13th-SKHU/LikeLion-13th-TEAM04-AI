# chat.py
# 역할 추출 + 추천 요약

from __future__ import annotations
import os, json, re, time, hashlib
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv
load_dotenv()  # .env 


USE_TOGETHER = os.getenv("USE_TOGETHER_CHAT", "false").lower() == "true"
MODEL_CHAT   = os.getenv("MODEL_CHAT", "meta-llama/Llama-3.1-8B-Instruct-Turbo")
API_KEY      = os.getenv("TOGETHER_API_KEY") or ""
CACHE_TTL    = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  

# Together 
client = None
if USE_TOGETHER and API_KEY:
    try:
        from together import Together
        client = Together(api_key=API_KEY)
    except Exception as e:
        print("[chat] Together import/init failed -> fallback only:", e)
        client = None


_cache: Dict[str, Tuple[float, Any]] = {}

def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _get_cache(k: str):
    v = _cache.get(k)
    if not v:
        return None
    exp, val = v
    if exp < time.time():
        _cache.pop(k, None)
        return None
    return val

def _set_cache(k: str, val: Any, ttl: int = CACHE_TTL):
    _cache[k] = (time.time() + ttl, val)

# 역할 힌트
ROLE_PATTERNS = [
    (re.compile(r"(상인|사장|사장님|점주|고용|모집|구함|찾아요|채용)"), "상인"),
    (re.compile(r"(청년|지원|일할|알바|파트타임|포트폴리오|할\s*수|가능)"), "청년"),
]

# 행정동/구 일부
REGION_PAT = re.compile(r"(마포구|종로구|성북구|강남구|용산구|관악구|강북구|송파구|구로구|영등포구|동작구|서대문구|노원구|중랑구|도봉구|서초구|성동구|중구|강동구|양천구|강서구|은평구)")

# 시간 패턴
TIME_PAT = re.compile(r"(주\s*\d+\s*회|평일|주말|오전|오후|저녁|야간|새벽)")

# 키, 동의어 
SKILL_CANON = {
    "카페 바리스타": [r"바리스타", r"라떼아트", r"에스프레소", r"브루잉", r"핸드드립", r"커피"],
    "포스터 디자인": [r"포스터", r"디자인", r"포토샵", r"인디자인", r"일러스트"],
    "스마트스토어 운영": [r"스마트스토어", r"상세페이지", r"상품\s*등록", r"CS", r"키워드\s*광고"],
    "영상 촬영/편집": [r"영상", r"촬영", r"편집", r"프리미어", r"쇼츠"],
    "사진 촬영": [r"사진", r"인물\s*촬영", r"보정"],
    "SNS 마케팅": [r"SNS", r"인스타그램", r"틱톡", r"해시태그", r"콘텐츠"],
    "웹 퍼블리셔": [r"퍼블리셔", r"HTML", r"CSS", r"부트스트랩"],
    "프론트엔드 개발": [r"프론트엔드", r"React", r"Vue", r"TypeScript"],
    "백엔드 개발": [r"백엔드", r"Flask", r"Node\.js", r"Django", r"FastAPI"],
    "데이터 라벨링": [r"라벨링", r"OCR", r"음성\s*QA"],
}

def _canon_skill(text: str) -> str:
    t = text.lower()
    for canon, pats in SKILL_CANON.items():
        for p in pats:
            if re.search(p, t, flags=re.I):
                return canon
    return ""

def _rule_extract(text: str) -> Tuple[Dict[str, str], bool]:
    role = ""
    for pat, lab in ROLE_PATTERNS:
        if pat.search(text):
            role = lab
            break

    regions = REGION_PAT.findall(text) or []
    times   = TIME_PAT.findall(text) or []
    skill   = _canon_skill(text)

    ctx = {
        "role": role,
        "region": " ".join(regions[:2]),
        "time": " ".join(times[:3]),
        "skills": skill,
    }
    confident = bool(role) and (ctx["region"] or ctx["time"] or ctx["skills"])
    return ctx, confident

# LLM json
def _safe_json_loads(txt: str) -> Optional[Dict[str, str]]:
    try:
        s = txt.strip().strip("`")
        s = re.sub(r"^\s*json\s*", "", s, flags=re.I)
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def normalize_query(user_text: str) -> Dict[str, str]:
    """
    사용자 한 문장 → {role, region, time, skills}
    - 먼저 규칙 기반 시도
    - 필요 시 Together Chat 호출 (USE_TOGETHER=true + client 유효)
    - 항상 dict로 반환 (키 보정)
    """
    user_text = (user_text or "").strip()
    key = "norm:" + _h(user_text)
    cached = _get_cache(key)
    if cached:
        return cached

    #  규칙 기반
    ctx, ok = _rule_extract(user_text)
    if ok or not USE_TOGETHER or client is None:
        _set_cache(key, ctx)
        return ctx

    # LLM 호출
    sys = (
        "너는 상인/청년 매칭 보조야. 사용자 입력에서 "
        "{role, region, time, skills}만 추출해 한국어 JSON으로만 출력해줘. "
        "예: {\"role\":\"상인\",\"region\":\"마포구\",\"time\":\"주 2회 오후\",\"skills\":\"포스터 디자인\"}"
    )
    try:
        r = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            max_tokens=180,
        )
        raw = r.choices[0].message.content or ""
        parsed = _safe_json_loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned non-JSON")

        out = {
            "role":   (parsed.get("role") or "").strip(),
            "region": (parsed.get("region") or "").strip(),
            "time":   (parsed.get("time") or "").strip(),
            "skills": (parsed.get("skills") or "").strip(),
        }
        # 스킬 캐논 보정(LLM이 애매하게 주면 정규화)
        if out["skills"] and out["skills"] not in SKILL_CANON:
            canon = _canon_skill(out["skills"])
            if canon:
                out["skills"] = canon

        _set_cache(key, out)
        return out
    except Exception as e:
        print("[normalize_query] LLM error -> fallback:", e)
        _set_cache(key, ctx)
        return ctx


def explain_recommendations(context: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    """
    상위 후보 3명 정도를 근거와 함께 간단 bullet로 요약
    - LLM 사용 가능하면 LLM 아니면 규칙 기반 bullet
    """
    top = [{"name": c.get("name"), "profile": c.get("profile")} for c in candidates[:3]]
    key = "rec:" + _h(json.dumps({"ctx": context, "top": top}, ensure_ascii=False))
    cached = _get_cache(key)
    if cached:
        return cached

    # LLM 미사용.불가=> 간단 bullet
    if not USE_TOGETHER or client is None:
        bullets = [f"- {c.get('name','이름미상')}: {c.get('profile','')}" for c in top if c.get("name")]
        text = "추천 후보:\n" + "\n".join(bullets) if bullets else "추천 후보가 없습니다."
        _set_cache(key, text)
        return text

    # LLM 사용
    sys = "요청과 후보 리스트를 보고 핵심 근거만 3~4줄 bullet로 한국어로 작성해줘. 군더더기는 금지야."
    prompt = f"요청: {context}\n후보: {top}"
    try:
        r = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        out = r.choices[0].message.content or ""
        _set_cache(key, out)
        return out
    except Exception as e:
        print("[explain_recommendations] LLM error -> fallback:", e)
        bullets = [f"- {c.get('name','이름미상')}: {c.get('profile','')}" for c in top if c.get("name")]
        text = "추천 후보:\n" + "\n".join(bullets) if bullets else "추천 후보가 없습니다."
        _set_cache(key, text)
        return text
