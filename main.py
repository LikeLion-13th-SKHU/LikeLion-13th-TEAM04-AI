# flask 


from flask import Flask, request, jsonify
from chat import normalize_query, explain_recommendations
from embedding_search import search_candidates

app = Flask(__name__)

#skills가 너무 안나와서 추가해봄 삭제할수도
import re

SKILL_SYNONYMS = {
    "카페 바리스타": ["바리스타", "커피", "에스프레소", "라떼아트", "브루잉", "핸드드립"],
    "포스터 디자인": ["포스터", "디자인", "일러스트", "포토샵", "인디자인"],
    "스마트스토어 운영": ["스마트스토어", "상세페이지", "상품 등록", "CS", "키워드 광고"],
    # 필요시 계속 추가
}

def _normalize_ko(s: str) -> str:
    return re.sub(r"[\s,./·|]+", "", s).lower()

def _kw_overlap_score(doc: str, skills_hint: str) -> float:
    if not skills_hint:
        return 0.0
    doc_n = _normalize_ko(doc)
    # 힌트 + 동의어 확장
    cands = [skills_hint] + SKILL_SYNONYMS.get(skills_hint, [])
    toks = set()
    for t in cands:
        for x in re.split(r"[\s,./·|]+", t.strip()):
            if x:
                toks.add(_normalize_ko(x))
    if not toks:
        return 0.0
    # 부분 일치 허용(substring)
    hits = sum(1 for t in toks if t in doc_n)
    return hits / max(1, len(toks))


@app.route("/chat", methods=["POST"])
def chat_normalize():
    user_text = request.json.get("text", "")
    ctx = normalize_query(user_text)
    return jsonify(ctx)
    #return {"message": "청상회 ai 챗봇"}

@app.route("/search", methods=["POST"])
def search():
    try:
        body = request.json or {}
        ctx = body.get("context", {}) or {}
        q_parts = [ctx.get("region","").strip(), ctx.get("time","").strip(), ctx.get("skills","").strip()]
        q = " ".join([p for p in q_parts if p]).strip() or (body.get("q") or "").strip()

        if not q:
            return jsonify({
                "error":"context or q required",
                "detail":{
                    "received_body": body,
                    "computed_q_parts": q_parts,
                    "hint": "context.region/time/skills 중 하나는 채우거나 q를 직접 보내주세요."
                }
            }), 400

        skills_hint = (ctx.get("skills") or body.get("skills") or "").strip()
        results = search_candidates(q, top_k=int(body.get("top_k", 5)), skills_hint=skills_hint)
        return jsonify({"query": q, "results": results})
    except Exception as e:
        import traceback
        print(">>> /search error\n", traceback.format_exc())
        return jsonify({"error":"internal_error","detail":str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    ctx = request.json.get("context", {})
    candidates = request.json.get("candidates", [])
    text = explain_recommendations(ctx, candidates)
    return jsonify({"summary": text})

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
