# OpenCV 파라미터 튜닝 프레임워크

이미지 내의 점(dot)을 효과적으로 검출하기 위한 최적의 OpenCV 필터 파이프라인과 파라미터를 찾는 자동화된 튜닝 프로젝트입니다.

## 주요 기능
- 체계적인 그리드 서치 기반의 파라미터 탐색
- 실험 제어판을 통한 손쉬운 파이프라인(필터 순서, 종류) 변경
- 대용량 조합 테스트를 위한 Iterator 및 Chunk 기반의 메모리 최적화
- Parquet 포맷을 사용한 효율적인 결과 로그 관리
- 중단/재시작이 가능한 체크포인트 기능
- Top-N 결과에 대한 자동 상세 분석 (단계별 이미지, 시각화 자료 생성)
- 파라미터 영향도, 상관관계 등을 포함한 종합 분석 리포트 자동 생성

## 디렉토리 구조
opencv_tuning/
├── data/
│   └── raw/          # 원본 데이터
├── results/          # 실행 결과 (gitignore 처리됨)
├── src/              # 소스 코드
└── ...
## 설치 방법
```bash
# 1. 가상환경 생성 및 활성화
python3.10 -m venv venv
source venv/bin/activate  # macOS/Linux
# .\\venv\\Scripts\\activate  # Windows

# 2. 필요 패키지 설치
pip install -r requirements.txt

## 실행방법

# src 폴더로 이동
cd src

# 이어하기 (기본)
python grid_search_analyzer.py

# 처음부터 새로 시작하기
python grid_search_analyzer.py --fresh-start