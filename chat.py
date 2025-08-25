

# chat_slim.py
# 규칙 기반 역할/지역/시간/스킬 추출 + 간단 요약 (LLM/캐시 제거)

from __future__ import annotations
import json, re
from typing import Dict, List, Tuple, Any, Optional

# 규칙 패턴
ROLE_PATTERNS = [
    (re.compile(r"(상인|사장|사장님|점주|고용|모집|구함|찾아요|채용)"), "상인"),
    (re.compile(r"(청년|지원|일할|알바|파트타임|포트폴리오|할\s*수|가능)"), "청년"),
]

REGION_PAT = re.compile(r"(마포구|종로구|성북구|강남구|용산구|관악구|강북구|송파구|구로구|영등포구|동작구|서대문구|노원구|중랑구|도봉구|서초구|성동구|중구|강동구|양천구|강서구|은평구)")
TIME_PAT   = re.compile(r"(주\s*\d+\s*회|평일|주말|오전|오후|저녁|야간|새벽)")
GENDER_PAT = re.compile(r"(남자|여자|남성|여성|남|여)")

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
    gender  = "".join(GENDER_PAT.findall(text))[:1]  # 하나만

    ctx = {
        "role": role,
        "region": " ".join(regions[:2]),
        "time": " ".join(times[:3]),
        "skills": skill,
        "gender": gender
    }
    confident = bool(role) and (ctx["region"] or ctx["time"] or ctx["skills"])
    return ctx, confident

def normalize_query(user_text: str) -> Dict[str, str]:
    user_text = (user_text or "").strip()
    ctx, _ = _rule_extract(user_text)
    return ctx

def explain_recommendations(context: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    top = [{"name": c.get("name"), "gender": c.get("gender","성별미상"), "profile": c.get("profile")} for c in candidates[:3]]
    bullets = [f"- {c.get('name','이름미상')} ({c.get('gender')}): {c.get('profile','')}" for c in top if c.get("name")]
    return "추천 후보:\n" + "\n".join(bullets) if bullets else "추천 후보가 없습니다."
