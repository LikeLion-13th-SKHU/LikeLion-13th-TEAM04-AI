#투게더 에이피아이-> 챗만 사용할 거임
#역할 추출이랑 추천 요약해주는 부분만!


import os, json, re, time, hashlib
from dotenv import load_dotenv
from together import Together

load_dotenv() #.env
USE_TOGETHER = os.getenv("USE_TOGETHER_CHAT", "true").lower() == "true"
MODEL_CHAT = os.getenv("MODEL_CHAT", "meta-llama/Llama-3.1-8B-Instruct-Turbo")
MODEL_FB   = os.getenv("MODEL_CHAT_FALLBACK", "meta-llama/Llama-3.3-70B-Instruct")
CACHE_TTL  = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# 아주 단순한 메모리 캐시 
_cache = {}  # key -> (expire_ts, value)
def _get_cache(k):
    v = _cache.get(k); 
    if not v: return None
    exp, val = v
    if exp < time.time(): _cache.pop(k, None); return None
    return val
def _set_cache(k, val, ttl=CACHE_TTL): _cache[k] = (time.time()+ttl, val)
def _h(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

# 규칙기반(무료) 1차 추출
ROLE_PATTERNS = [
    (re.compile(r"(사장|사장님|점주|고용|구함|찾아요|채용)"), "구인자"),
    (re.compile(r"(일할|알바|구직|지원|포트폴리오|할 수|가능)"), "구직자"),
]

def _rule_extract(text: str):
    role = None
    for pat, lab in ROLE_PATTERNS:
        if pat.search(text): role = lab; break
    region = re.findall(r"(마포구|구로구|성북구|관악구|영등포구|동작구|서울)", text)
    time_  = re.findall(r"(주\s*\d+회|오전|오후|저녁|주말|평일)", text)
    skills = re.findall(r"(포스터 디자인|포토샵|스마트스토어|SNS|영상|촬영|마케팅)", text)
    ctx = {"role": role or "", "region": " ".join(region[:2]),"time": " ".join(time_[:3]), "skills": " ".join(skills[:3])}
    confident = (role is not None) and (ctx["region"] or ctx["time"] or ctx["skills"])
    return ctx, confident

def normalize_query(user_text: str) -> dict:
    """ 사용자 입력 → {role, region, time, skills} (가능하면 무료 규칙, 아니면 Together Chat) """
    key = "norm:"+_h(user_text); hit = _get_cache(key)
    if hit: return hit

    ctx, ok = _rule_extract(user_text)
    if not USE_TOGETHER or ok:
        _set_cache(key, ctx); return ctx

    sys = ("너는 구인/구직 매칭 보조다. 사용자 입력에서 {role, region, time, skills}만 뽑아 " "JSON으로만 출력해. 예: {\"role\":\"구인자\",\"region\":\"마포구\",\"time\":\"주2회 오후\",\"skills\":\"포스터 디자인\"}")
    def _call(model):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys},{"role":"user","content":user_text}],
            temperature=0.1, max_tokens=180
        )
        txt = r.choices[0].message.content.strip().strip("`").replace("json","")
        return json.loads(txt)

    try: out = _call(MODEL_CHAT)
    except Exception: out = _call(MODEL_FB)

    _set_cache(key, out); return out

def explain_recommendations(context: dict, candidates: list) -> str:
    """ 상위 후보 3명 근거 요약 (짧게) """
    top = [{"name": c.get("name"), "profile": c.get("profile")} for c in candidates[:3]]
    key = "rec:"+_h(json.dumps({"c":top, "ctx":context}, ensure_ascii=False)); hit = _get_cache(key)
    if hit: return hit

    if not USE_TOGETHER:
        bullets = [f"- {c['name']}: {c['profile']}" for c in top]
        text = "추천 후보:\n" + "\n".join(bullets)
        _set_cache(key, text); return text

    sys = "요청과 후보 리스트를 보고 핵심 근거만 3~4줄 bullet로 한국어로 작성하라. 군더더기 금지."
    prompt = f"요청: {context}\n후보: {top}"
    r = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
        temperature=0.2, max_tokens=180
    )
    out = r.choices[0].message.content
    _set_cache(key, out); return out
