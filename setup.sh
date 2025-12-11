#!/bin/bash
#
# MLOps Chatbot Project - Setup Script
# 가상환경 생성 및 패키지 설치를 자동화합니다.
#

set -e  # 에러 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 헤더 출력
echo "============================================================"
echo "  MLOps Chatbot Project - Setup"
echo "============================================================"
echo ""

# Python 버전 확인
log_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

log_info "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    log_error "Python 3.10 or higher required (found $PYTHON_VERSION)"
    exit 1
fi

log_success "Python version check passed"
echo ""

# 가상환경 생성
if [ -d "venv" ]; then
    log_warning "Virtual environment already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf venv
    else
        log_info "Using existing virtual environment"
    fi
fi

if [ ! -d "venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists"
fi
echo ""

# 가상환경 활성화
log_info "Activating virtual environment..."
source venv/bin/activate
log_success "Virtual environment activated"
echo ""

# pip 업그레이드
log_info "Upgrading pip..."
pip install --upgrade pip --quiet
log_success "pip upgraded"
echo ""

# 패키지 설치 옵션
log_info "Choose installation mode:"
echo "  1) Full installation (all packages)"
echo "  2) Minimal installation (core packages only)"
echo "  3) Skip package installation"
read -p "Enter choice (1-3): " INSTALL_MODE

case $INSTALL_MODE in
    1)
        log_info "Installing all packages from requirements.txt..."
        pip install -r requirements.txt
        log_success "All packages installed"
        ;;
    2)
        log_info "Installing core packages only..."
        pip install torch transformers accelerate peft bitsandbytes
        pip install datasets pandas numpy
        pip install mlflow fastapi uvicorn
        pip install gradio jupyter
        pip install python-dotenv tqdm requests
        log_success "Core packages installed"
        ;;
    3)
        log_info "Skipping package installation"
        ;;
    *)
        log_warning "Invalid choice, skipping installation"
        ;;
esac
echo ""

# .env 파일 체크
if [ ! -f ".env" ]; then
    log_warning ".env file not found"
    log_info "Creating .env from .env.example..."
    cp .env.example .env
    log_success ".env file created"
    log_warning "Please edit .env file to add your API keys"
else
    log_info ".env file already exists"
fi
echo ""

# GPU 환경 체크
log_info "Checking GPU environment..."
python3 src/check_gpu.py
echo ""

# 완료 메시지
echo "============================================================"
log_success "Setup completed!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Edit .env file with your API keys:"
echo "     nano .env"
echo ""
echo "  3. Run GPU check:"
echo "     python src/check_gpu.py"
echo ""
echo "  4. Start with Phase 1:"
echo "     python src/01_test_base_model.py"
echo ""
echo "============================================================"
