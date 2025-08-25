

from flask import Flask, request, jsonify
from chat import normalize_query, explain_recommendations
from embedding_search import search_candidates

app = Flask(__name__)
user_states = {}  # state_key -> {"step":1|2, "ctx":{}}

def _merge_ctx(base: dict, newbits: dict) -> dict:
    out = dict(base or {})
    for k in ("role", "region", "time", "skills", "gender"):
        v = (newbits.get(k) or "").strip()
        if v: out[k] = v
    return out

def _role_to_target_type(role: str):
    return "청년" if role == "상인" else ("상인" if role == "청년" else None)

def _wrap(success: bool, code: str, message: str, data: dict):
    return {"success": success, "code": code, "message": message, "data": data}

def _simplify_results(results: list):
    return [{
        "id": r.get("id"), "name": r.get("name"), "profile": r.get("profile"),
        "score": r.get("score"), "type": r.get("type"),
        "gender": r.get("gender",""), "job": r.get("job"), "skills": r.get("skills")
    } for r in results]

@app.get("/healthz")
def health():
    return jsonify({"ok": True})

@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    room_id = body.get("roomId") or body.get("room_id")
    user_id = str(body.get("userId") if body.get("userId") is not None else body.get("user_id") or "anon").strip()
    text    = (body.get("text") or body.get("message") or "").strip()
    inc_ctx = body.get("context", {}) or {}

    state_key = f"{room_id}:{user_id}" if room_id is not None else user_id
    state = user_states.get(state_key, {"step": 1, "ctx": {}})
    ctx = _merge_ctx(state.get("ctx", {}), inc_ctx)

    try:
        # Step 1: 역할
        if state["step"] == 1:
            if text:
                ctx = _merge_ctx(ctx, normalize_query(text))
            role = (ctx.get("role") or "").strip()
            if role not in ("상인", "청년"):
                user_states[state_key] = {"step": 1, "ctx": ctx}
                return jsonify(_wrap(True, "OK", "역할 질문", {
                    "roomId": room_id, "userId": body.get("userId"),
                    "step": 1, "done": False,
                    "reply": "상인입니까, 청년입니까? (정확히 입력해주세요)",
                    "results": []
                })), 200

            user_states[state_key] = {"step": 2, "ctx": ctx}
            first_q = ("어떤 청년을 찾고 계신가요?\n예: '카페 바리스타, 마포구, 주 2회 오후'"
                        if role == "상인" else
                        "어떤 일을 할 수 있고 어디서 언제 일하고 싶으신가요?\n예: '포스터 디자인 가능, 성북구, 주 2회 오후'")
            return jsonify(_wrap(True, "OK", "조건 질문", {
                "roomId": room_id, "userId": body.get("userId"),
                "step": 2, "done": False, "reply": first_q, "results": []
            })), 200

        # Step 2: 조건 → 추천
        if state["step"] == 2:
            if text:
                ctx = _merge_ctx(ctx, normalize_query(text))

            for need, msg in (("region", "어느 지역에서 찾나요? (예: 마포구)"),
                                ("time", "가능 시간대를 알려주세요. (예: 주 2회 오후)"),
                                ("skills", "직무/스킬을 알려주세요. (예: 카페 바리스타)")):
                if not ctx.get(need):
                    user_states[state_key] = {"step": 2, "ctx": ctx}
                    return jsonify(_wrap(True, "OK", "조건 질문", {
                        "roomId": room_id, "userId": body.get("userId"),
                        "step": 2, "done": False, "reply": msg, "results": []
                    })), 200

            q = " ".join([ctx.get("region",""), ctx.get("time",""), ctx.get("skills","")]).strip()
            target_type = _role_to_target_type(ctx.get("role",""))
            top_k = int(body.get("top_k", 3))
            w_embed = float(body.get("w_embed", 0.4))
            w_kw = float(body.get("w_kw", 0.6))

            results = search_candidates(q, top_k=top_k, skills_hint=ctx.get("skills",""),
                                        w_embed=w_embed, w_kw=w_kw, target_type=target_type)
            summary = explain_recommendations(ctx, results)

            # 세션 종료 시 올바른 키로 초기화 (버그 수정: state_key 사용)
            user_states[state_key] = {"step": 1, "ctx": {}}

            return jsonify(_wrap(True, "OK", "추천 완료", {
                "roomId": room_id, "userId": body.get("userId"),
                "step": 3, "done": True, "reply": summary, "context": ctx,
                "results": _simplify_results(results)
            })), 200

        # 그 외 → 리셋
        user_states[state_key] = {"step": 1, "ctx": {}}
        return jsonify(_wrap(True, "RESET", "세션 초기화", {
            "roomId": room_id, "userId": body.get("userId"),
            "step": 1, "done": False, "reply": "세션을 초기화했어요. 상인입니까, 청년입니까?", "results": []
        })), 200

    except Exception as e:
        import traceback
        print(">>> /chat error\n", traceback.format_exc())
        print(">>> body:", body)
        print(">>> state:", user_states.get(state_key))
        return jsonify(_wrap(False, "INTERNAL_ERROR", str(e), {
            "roomId": room_id, "userId": body.get("userId")
        })), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
