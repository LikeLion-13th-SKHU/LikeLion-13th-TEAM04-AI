# 임베딩
#임베딩 기반 검색
# 각 json을 읽어-> 임베딩 인덱스(크로마디비)를 만들고-> 의미 유사도(E5임베딩) + 키워드 보너스를 합산해 사우이 후보 반환
# 모델: intfloat/multilingual-e5-large를 기본 사용
#저장소: chromadb.PersistentClient로 영속 컬렉션을 유지


from __future__ import annotations

import json, os, re, hashlib
from typing import Any, Dict, List
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# 경로/모델
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_data")
MODEL_NAME  = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large")
COLLECTION  = os.getenv("COLLECTION_NAME", "candidates")
DATA_DIR    = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

# 간단 동의어/토큰화
SKILL_SYNONYMS: Dict[str, List[str]] = {
    "카페 바리스타": ["바리스타", "커피", "에스프레소", "라떼아트", "브루잉", "핸드드립"],
    "포스터 디자인": ["포스터", "디자인", "일러스트", "포토샵", "인디자인"],
    "스마트스토어 운영": ["스마트스토어", "상세페이지", "상품 등록", "CS", "키워드 광고"],
}
_DELIMS = r"[\s,./·|]+"
_GENDER_PAT = re.compile(r"(남성|여성|남자|여자|남|여)")

def _normalize_ko(s: str) -> str:
    return re.sub(_DELIMS, "", (s or "")).lower()

def _tokenize(s: str) -> List[str]:
    return [t for t in re.split(_DELIMS, (s or "").strip()) if t]

def _expanded_terms(hint: str) -> List[str]:
    if not hint: return []
    terms = set(_tokenize(hint))
    n_hint = _normalize_ko(hint)
    for key, syns in SKILL_SYNONYMS.items():
        nk = _normalize_ko(key)
        if nk in n_hint or n_hint in nk:
            for s in syns: terms.update(_tokenize(s))
    return sorted(terms)

def _kw_overlap_score(doc: str, skills_hint: str, gender_hint: str = "") -> float:
    doc_n = _normalize_ko(doc)
    toks: List[str] = []
    if skills_hint: toks.extend(_expanded_terms(skills_hint))
    if gender_hint: toks.extend(_tokenize(gender_hint))
    toks = [t for t in toks if t]
    if not toks: return 0.0
    hits = sum(1 for t in toks if _normalize_ko(t) in doc_n)
    return hits / max(1, len(toks))

# 데이터 지문으로 인덱싱 필요시만 재구축
def _data_fingerprint() -> str:
    parts: List[str] = []
    for fn in ("seekers.json", "employers.json"):
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            st = os.stat(p)
            parts.append(f"{fn}:{st.st_size}:{int(st.st_mtime)}")
        else:
            parts.append(f"{fn}:missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

_side_meta = os.path.join(PERSIST_DIR, f"{COLLECTION}.version.json")

def _save_version(fp: str, count: int) -> None:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(_side_meta, "w", encoding="utf-8") as f:
        json.dump({"fp": fp, "count": count}, f, ensure_ascii=False)

def _load_version() -> Dict[str, Any]:
    if not os.path.exists(_side_meta): return {}
    try:
        with open(_side_meta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

_client = PersistentClient(path=PERSIST_DIR)
_col = _client.get_or_create_collection(COLLECTION)
_model = SentenceTransformer(MODEL_NAME)

def _load_all_profiles() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for fn in ("seekers.json", "employers.json"):
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path): continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        typ = "청년" if "seekers" in fn else "상인"
        for it in data:
            name = (it.get("name") or "noname").strip()
            gender = (it.get("gender") or "").strip()
            skills = (it.get("skills") or it.get("job") or "").strip()
            job    = (it.get("job") or it.get("skills") or "").strip()
            profile_only = (it.get("profile") or "").strip()
            merged = ", ".join([p for p in (skills or job, profile_only) if p])
            if merged:
                items.append({
                    "name": name, "type": typ, "gender": gender,
                    "job": job, "skills": skills,
                    "profile": merged, "profile_raw": profile_only,
                })
    return items

def _rebuild_index(items: List[Dict[str, str]]) -> None:
    global _col
    try: _client.delete_collection(COLLECTION)
    except Exception: pass
    _col = _client.get_or_create_collection(COLLECTION)

    ids   = [f"cand_{i}" for i in range(len(items))]
    docs  = [f"passage: {it['profile']}" for it in items]
    metas = [{
        "name": it["name"], "type": it["type"], "gender": it.get("gender",""),
        "job": it.get("job",""), "skills": it.get("skills",""),
        "profile_raw": it.get("profile_raw",""),
    } for it in items]

    vecs = _model.encode(docs, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    _col.add(ids=ids, documents=docs, embeddings=vecs.tolist(), metadatas=metas)

def _ensure_index() -> None:
    fp = _data_fingerprint()
    ver = _load_version()
    count = 0
    try: count = _col.count()
    except Exception: count = 0
    if count == 0 or ver.get("fp") != fp:
        items = _load_all_profiles()
        if not items: return
        _rebuild_index(items)
        _save_version(fp, len(items))

def search_candidates(
    query_text: str,
    top_k: int = 5,
    skills_hint: str = "",
    w_embed: float = 0.4,
    w_kw: float = 0.6,
    target_type: str | None = None,
) -> List[Dict[str, Any]]:
    _ensure_index()
    try:
        if _col.count() == 0: return []
    except Exception:
        return []

    try:
        top_k = int(top_k)
        if top_k <= 0: top_k = 5
    except Exception:
        top_k = 5

    q_text = (query_text or "").strip()
    gmatch = _GENDER_PAT.findall(q_text)
    gender_hint = " ".join(gmatch[:1])

    boosted = " ".join([q_text] + ([skills_hint.strip()] * 4 if skills_hint else []))
    q_emb = _model.encode([f"query: {boosted}"], convert_to_numpy=True, normalize_embeddings=True)[0]

    res = _col.query(query_embeddings=[q_emb.tolist()], n_results=max(top_k * 10, 50), include=["documents", "metadatas", "distances"])
    ids   = (res.get("ids") or [[]])[0]
    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs):
        doc_plain = (doc or "").replace("passage: ", "", 1).strip()
        embed_score = float(1 - dists[i]) if i < len(dists) else 0.0
        kw_score = _kw_overlap_score(doc_plain, skills_hint, gender_hint=gender_hint)
        final = w_embed * embed_score + w_kw * kw_score
        meta = metas[i] if i < len(metas) and metas[i] else {}
        out.append({
            "id": ids[i] if i < len(ids) else f"cand_{i}",
            "name": meta.get("name"), "type": meta.get("type"),
            "gender": meta.get("gender",""), "job": meta.get("job",""),
            "skills": meta.get("skills",""), "profile": doc_plain,
            "score": final, "debug": {"embed": embed_score, "kw": kw_score},
        })

    if target_type in ("청년", "상인"):
        out = [r for r in out if r.get("type") == target_type]

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


