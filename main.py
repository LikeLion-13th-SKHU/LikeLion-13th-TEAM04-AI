# flask

from flask import Flask, request, jsonify
from chat import normalize_query, explain_recommendations
from embedding_search import search_candidates

app = Flask(__name__)


user_states = {}  # { user_id: {"step": 1|2, "ctx": {...}} }

def _merge_ctx(base: dict, newbits: dict) -> dict:
    out = dict(base or {})
    for k in ("role", "region", "time", "skills"):
        v = (newbits.get(k) or "").strip()
        if v:
            out[k] = v
    return out

def _role_to_target_type(role: str):
    return "청년" if role == "상인" else ("상인" if role == "청년" else None)


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    user_id = (body.get("user_id") or "anon").strip()
    message = (body.get("message") or "").strip()
    inc_ctx = body.get("context", {}) or {}

    try:
        # 상태 로드/초기화
        state = user_states.get(user_id, {"step": 1, "ctx": {}})
        ctx = _merge_ctx(state.get("ctx", {}), inc_ctx)

        # 역할 수집
        if state["step"] == 1:
            if message:
                parsed = normalize_query(message) or {}
                ctx = _merge_ctx(ctx, parsed)

            role = (ctx.get("role") or "").strip()
            if role not in ("상인", "청년"):
                user_states[user_id] = {"step": 1, "ctx": ctx}
                return jsonify({"reply": "상인입니까, 청년입니까? (정확히 입력해주세요)"}), 200

            user_states[user_id] = {"step": 2, "ctx": ctx}
            if role == "상인":
                return jsonify({"reply": "어떤 청년을 찾고 계신가요?\n예: '카페 바리스타, 마포구, 주 2회 오후'"}), 200
            else:
                return jsonify({"reply": "어떤 일을 할 수 있고 어디서 언제 일하고 싶으신가요?\n예: '포스터 디자인 가능, 성북구, 주 2회 오후'"}), 200

        # 조건 수집 -> 추천
        if state["step"] == 2:
            if message:
                parsed = normalize_query(message) or {}
                ctx = _merge_ctx(ctx, parsed)

            # 추가 질문(필요하면)
            if not ctx.get("region"):
                user_states[user_id] = {"step": 2, "ctx": ctx}
                return jsonify({"reply": "어느 지역에서 찾나요? (예: 마포구)"}), 200
            if not ctx.get("time"):
                user_states[user_id] = {"step": 2, "ctx": ctx}
                return jsonify({"reply": "가능 시간대를 알려주세요. (예: 주 2회 오후)"}), 200
            if not ctx.get("skills"):
                user_states[user_id] = {"step": 2, "ctx": ctx}
                return jsonify({"reply": "직무/스킬을 알려주세요. (예: 카페 바리스타)"}), 200
            if not ctx.get("gender"):
                user_states[user_id] = {"step": 2, "ctx": ctx}
                return jsonify({"reply": "성별을 알려주세요. (예: 남성, 여성)"}), 200

            # 검색 질의
            q = " ".join([ctx.get("region",""), ctx.get("time",""), ctx.get("skills","")]).strip()
            target_type = _role_to_target_type(ctx.get("role",""))
            top_k = int(body.get("top_k", 3))
            w_embed = float(body.get("w_embed", 0.4))
            w_kw = float(body.get("w_kw", 0.6))
            ctx.get("gender", "")

            results = search_candidates(
                q,
                top_k=top_k,
                skills_hint=ctx.get("skills",""),
                w_embed=w_embed,
                w_kw=w_kw,
                target_type=target_type
            )
            summary = explain_recommendations(ctx, results)

            # 세션 종료
            user_states[user_id] = {"step": 1, "ctx": {}}

            return jsonify({"reply": summary, "results": results, "context_used": ctx}), 200

        # 비정상 상태면 초기화
        user_states[user_id] = {"step": 1, "ctx": {}}
        return jsonify({"reply": "세션을 초기화했어요. 상인입니까, 청년입니까?"}), 200

    except Exception as e:
        import traceback
        print(">>> /chat error\n", traceback.format_exc())
        print(">>> body:", body)
        print(">>> state:", user_states.get(user_id))
        return jsonify({"error": "internal_error", "detail": str(e)}), 500


@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
