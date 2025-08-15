# 임베딩
#임베딩 기반 검색



# 임베딩 기반 검색 

import os, re, json
from typing import List, Dict, Any
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_data")

HERE = os.path.dirname(__file__)
CANDIDATE_DIRS = [os.path.join(HERE, "data"), os.path.join(os.path.dirname(HERE), "data")]

def _find_data_dir():
    for d in CANDIDATE_DIRS:
        if os.path.isdir(d):
            return d
    return None

DATA_DIR = _find_data_dir()
print("[embedding_search] CWD =", os.getcwd())
print("[embedding_search] DATA_DIR =", DATA_DIR)

_client = PersistentClient(path=PERSIST_DIR)
_col = _client.get_or_create_collection("candidates")
_model = SentenceTransformer("intfloat/multilingual-e5-large")


def _load_all_profiles() -> List[Dict[str, str]]:
    """seekers.json, employers.json에서 profile 문장 로드"""
    if not DATA_DIR:
        print("[embedding_search] data dir not found")
        return []
    items: List[Dict[str, str]] = []
    for fn in ("seekers.json", "employers.json"):
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for it in data:
                    prof = (it.get("profile") or it.get("desc") or "").strip()
                    name = (it.get("name") or "noname").strip()
                    if prof:
                        items.append({"name": name, "profile": prof})
            except Exception as ex:
                print("[embedding_search] JSON load error:", ex)
    print(f"[embedding_search] loaded profiles: {len(items)}")
    return items


def _ensure_index():
    """컬렉션 비어 있으면 인덱스 생성. E5 권장 포맷(passsage:) 적용"""
    try:
        if _col.count() > 0:
            return
    except Exception as ex:
        print("[embedding_search] col.count error:", ex)

    items = _load_all_profiles()
    if not items:
        print("[embedding_search] no items to index")
        return

    ids = [f"cand_{i}" for i in range(len(items))]
    
    docs = [f"passage: {it['profile']}" for it in items]
    metas = [{"name": it["name"]} for it in items]

    vecs = _model.encode(
        docs,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    _col.add(ids=ids, documents=docs, embeddings=vecs.tolist(), metadatas=metas)
    print(f"[embedding_search] indexed {len(ids)} docs")


def _kw_overlap_score(doc: str, skills_hint: str) -> float:
    """아주 단순한 키워드 일치 보너스 (0~1)"""
    if not skills_hint:
        return 0.0
    toks = [t for t in re.split(r"[\s,./·|]+", skills_hint.strip()) if t]
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if t in doc)
    return hit / max(1, len(toks))


def search_candidates(query_text: str, top_k: int = 5, skills_hint: str = "",
                    w_embed: float = 0.4, w_kw: float = 0.6,  # ← 스킬 가중 강화
                    target_type=None):
    """
    임베딩 검색 + 스킬 가중:
        - E5 query/passage 접두어
        - skills_hint 2회 반복으로 가중 강화
        - 임베딩 점수 + 키워드 보너스 가중합으로 최종 정렬
    """
    _ensure_index()

    # 컬렉션 상태 확인
    try:
        if _col.count() == 0:
            return []
    except Exception as ex:
        print("[embedding_search] col.count after ensure error:", ex)
        return []

    # top_k 가드
    try:
        top_k = int(top_k)
        if top_k <= 0:
            top_k = 5
    except Exception:
        top_k = 5

    
    q_text = query_text.strip()
    if skills_hint:
        boosted = " ".join([q_text] + [skills_hint.strip()] * 4)  # ← 4회 반복
    else:
        boosted = q_text
    q_text_prefixed = f"query: {boosted}".strip()

    q_emb = _model.encode(
        [q_text_prefixed],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # include에 'ids' 넣지X
    res = _col.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 안전 가드로 꺼내기
    ids = (res.get("ids") or [[]])[0] if res.get("ids") else []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(len(docs)):
        embed_score = float(1 - dists[i]) if i < len(dists) else 0.0
        # passage: 접두어 제거한 원문을 보여주고 싶다면:
        doc_plain = docs[i].removeprefix("passage: ").strip()
        kw_score = _kw_overlap_score(doc_plain, skills_hint)
        final = w_embed * embed_score + w_kw * kw_score
        out.append({
            "id": ids[i] if i < len(ids) else f"cand_{i}",
            "name": (metas[i] or {}).get("name") if i < len(metas) else None,
            "profile": doc_plain,
            "score": final,
            "debug": {"embed": embed_score, "kw": kw_score}  # 배포 때 제거 가능
        })

    # 최종 점수 기준 재정렬
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
