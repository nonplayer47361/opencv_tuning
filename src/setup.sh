#!/bin/bash

echo "=================================================="
echo "  OpenCV Tuning 프로젝트 환경 설정을 시작합니다. (macOS/Linux)  "
echo "=================================================="

# Python 3.10 버전 확인
if ! command -v python3.10 &> /dev/null
then
    echo "오류: python3.10을 찾을 수 없습니다."
    echo "Python 3.10을 먼저 설치해주세요. (pyenv 사용 권장)"
    exit 1
fi

echo "Python 3.10을 확인했습니다."

# 가상환경 생성
echo "\n[1/3] 'venv' 이름으로 가상환경을 생성합니다..."
python3.10 -m venv venv
if [ $? -ne 0 ]; then
    echo "오류: 가상환경 생성에 실패했습니다."
    exit 1
fi

# 가상환경 활성화 및 패키지 설치
echo "\n[2/3] 가상환경을 활성화하고 필요한 패키지를 설치합니다..."
source venv/bin/activate
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "오류: 패키지 설치에 실패했습니다."
    exit 1
fi

# 필수 폴더 구조 생성
echo "\n[3/3] 필수 데이터 및 소스 폴더를 생성합니다..."
mkdir -p data/raw
mkdir -p src

echo "\n=================================================="
echo "✅ 모든 설정이 완료되었습니다!"
echo "이제 아래 순서대로 프로젝트를 진행하세요."
echo ""
echo "  1. data/raw/ 폴더에 이미지와 csv 파일을 넣으세요."
echo "  2. src/ 폴더에 메인 파이썬 스크립트를 넣으세요."
echo "  3. 터미널을 새로 열고 'source venv/bin/activate' 명령어로 가상환경을 활성화한 뒤 사용하세요."
echo "=================================================="