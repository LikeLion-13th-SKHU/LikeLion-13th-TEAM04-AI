# 이건 데이터 json파일로 자동 저장됨.
# 여기서 수정하고 실행하면 data 파일에 있는 json폴더에 자동으로 수정됨.!!
import json
import random
import os

# 저장 경로
os.makedirs("data", exist_ok=True)

# 샘플 데이터
regions = ["서울 마포구", "서울 강남구", "서울 성북구", "서울 용산구", "서울 구로구", "서울 서초구", "서울 송파구", "서울 강북구", "서울 종로구", "서울 관악구"]

times = ["평일 오전", "평일 오후", "주말 오전", "주말 오후", "주 2회 오후", "주 3회 오전"]

skills_seekers = {
    "카페 바리스타": ["라떼아트 가능", "브루잉 가능", "디저트 제조 가능"],
    "스마트스토어 운영": ["상품 등록", "상세페이지 제작", "고객 응대"],
    "영상 촬영/편집": ["프리미어 편집", "쇼츠 제작", "인터뷰 촬영"],
    "SNS 마케팅": ["인스타그램 운영", "해시태그 전략", "틱톡 콘텐츠 제작"],
    "포스터 디자이너": ["포토샵 가능", "일러스트 가능", "인디자인 가능"],
    "사진 촬영": ["제품 촬영", "인물 촬영", "보정 가능"],
    "웹 퍼블리셔": ["반응형 웹", "HTML/CSS", "부트스트랩"],
    "프론트엔드 개발": ["React", "Vue", "TypeScript"],
    "백엔드 개발": ["Flask", "Node.js", "Django"],
    "데이터 라벨링": ["이미지 라벨링", "OCR 교정", "음성 QA"]
}

skills_employers = {
    "카페 바리스타": ["주말 오후 근무", "라떼아트 가능자", "경력 6개월 이상"],
    "스마트스토어 운영": ["상품 등록 경험자", "데이터 분석 가능자", "광고 세팅 가능자"],
    "영상 촬영/편집": ["홍보 영상 제작", "숏폼 제작 경험자", "인터뷰 촬영 가능자"],
    "SNS 마케팅": ["콘텐츠 기획 가능자", "틱톡 운영 경험자", "블로그 운영 가능자"],
    "포스터 디자이너": ["행사 포스터 제작", "브랜딩 경험자", "인디자인 가능자"],
    "사진 촬영": ["제품 촬영 경험자", "인물 촬영 가능자", "보정 가능자"],
    "웹 퍼블리셔": ["반응형 웹 작업 가능자", "접근성 작업 가능자", "Figma 퍼블리싱 가능자"],
    "프론트엔드 개발": ["React 가능자", "Vue 가능자", "TypeScript 가능자"],
    "백엔드 개발": ["Flask API 개발 가능자", "Node.js 가능자", "Django 가능자"],
    "데이터 라벨링": ["이미지 라벨링 가능자", "OCR 교정 가능자", "음성 데이터 QA 가능자"]
}

# 이름 생성 함수
def generate_name():
    last_names = ["김", "이", "박", "최", "정", "한", "오", "서", "신", "윤"]
    first_names = ["지민", "수현", "도윤", "서연", "민준", "하늘", "예진", "현우", "지연", "다영"]
    return random.choice(last_names) + random.choice(first_names)

# 구직자 데이터 생성
seekers = []
for job, skills in skills_seekers.items():
    for _ in range(3):  # 각 직업별 3명 이상
        seekers.append({
            "name": generate_name(),
            "job": job,
            "profile": f"{random.choice(regions)}, {random.choice(times)}, {random.choice(skills)}"
        })

# 구인자 데이터 생성
employers = []
for job, requirements in skills_employers.items():
    for _ in range(3):
        employers.append({
            "name": generate_name() + " 사장님",
            "job": job,
            "profile": f"{random.choice(regions)}, {random.choice(times)}, {random.choice(requirements)}"
        })

# JSON 저장
with open("data/seekers.json", "w", encoding="utf-8") as f:
    json.dump(seekers, f, ensure_ascii=False, indent=2)

with open("data/employers.json", "w", encoding="utf-8") as f:
    json.dump(employers, f, ensure_ascii=False, indent=2)

print("seekers.json, employers.json 생성 완료!")
