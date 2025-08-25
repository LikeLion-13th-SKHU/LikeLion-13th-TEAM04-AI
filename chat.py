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

REGION_PAT = re.compile(r"(종로구|중구|용산구|성동구|광진구|동대문구|중랑구|성북구|강북구|도봉구|노원구|은평구|서대문구|마포구|양천구|강서구|구로구|금천구|영등포구|동작구|관악구|서초구|강남구|송파구|강동구|중구|서구|동구|영도구|부산진구|동래구|남구|북구|해운대구|사하구|금정구|강서구|연제구|수영구|사상구|중구|동구|서구|남구|북구|수성구|달서구|중구|동구|미추홀구|연수구|남동구|부평구|계양구|서구|동구|서구|남구|북구|광산구|동구|중구|서구|유성구|대덕구|중구|남구|동구|북구)")
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
    "요식업/주방":   [r"주방보조", r"주방", r"조리", r"설거지", r"식당", r"홀서빙", r"레스토랑", r"키친"],
    "편의점/마트":   [r"편의점", r"마트", r"캐셔", r"계산대", r"진열", r"발주", r"재고"],
    "물류/창고":     [r"물류", r"창고", r"택배", r"상하차", r"분류", r"포장", r"피킹|패킹", r"지게차"],
    "이사/배송":     [r"이사", r"용달", r"퀵", r"배달", r"배송"],
    "고객 상담/콜센터": [r"(?:콜센터|고객\s*센터|CS)", r"고객\s*응대", r"인바운드", r"아웃바운드", r"채팅\s*상담"],
    "행정 사무":     [r"행정", r"사무", r"문서", r"보고서", r"엑셀", r"오피스"],
    "디지털 마케팅": [r"GA4?|애널리틱스", r"광고", r"캠페인", r"CRM", r"퍼포먼스", r"세팅"],
    "그래픽 디자인": [r"브랜딩", r"편집\s*디자인", r"시각\s*디자인", r"CI|BI"],
    "UX/UI 디자이너":[r"Figma", r"프로토타입", r"와이어프레임", r"디자인\s*시스템"],
    "영상 촬영/편집": [r"영상", r"촬영", r"편집", r"프리미어|Premiere", r"애프터\s*이펙츠|After\s*Effects|AE", r"쇼츠|Shorts", r"유튜브|YouTube", r"브이로그|Vlog"],
    "AI/데이터 분석": [r"데이터\s*분석", r"머신러닝|ML", r"딥러닝|DL", r"\bAI\b", r"통계", r"파이썬|Python", r"\bR\b", r"SQL"],
    "인사/HR":       [r"채용", r"면접", r"HRD", r"조직", r"HR", r"교육"],
    "회계/재무":     [r"회계", r"재무", r"세무", r"더존", r"ERP", r"부가세", r"결산"],
    "총무":          [r"비품", r"자산", r"계약", r"행사", r"문서\s*관리"],
    "교육/강사":     [r"강사", r"교사", r"튜터", r"과외", r"수업", r"교안"],
    "번역/통역":     [r"번역", r"통역", r"영한|한영", r"중국어|중한|한중", r"일본어|일한|한일"],
    "가사/돌봄":     [r"청소", r"가사", r"도우미", r"베이비시터", r"아이\s*돌봄", r"펫\s*시터|반려동물"],
    "보안":          [r"침투\s*테스트", r"방화벽", r"네트워크\s*보안", r"시큐리티|Security"],
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
