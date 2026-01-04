#!/bin/bash

# Amazon Nova 2 Omni Streamlit 데모 실행 스크립트

echo "🤖 Amazon Nova 2 Omni Streamlit 데모를 시작합니다..."

# uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "❌ uv가 설치되지 않았습니다. 다음 명령어로 설치하세요:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 의존성 설치
echo "📦 uv로 의존성을 설치합니다..."
uv sync

# AWS 자격증명 확인
echo "🔐 AWS 자격증명을 확인합니다..."
aws sts get-caller-identity > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ AWS 자격증명이 설정되어 있습니다."
else
    echo "❌ AWS 자격증명이 설정되지 않았습니다. 다음 명령어로 설정하세요:"
    echo "aws configure"
    exit 1
fi

# Streamlit 앱 실행
echo "🚀 Streamlit 데모를 시작합니다..."
uv run streamlit run main.py --server.port 8501