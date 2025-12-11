# MLOps Chatbot Project - Quick Start Guide

이 가이드는 프로젝트를 빠르게 시작하는 방법을 단계별로 안내합니다.

## 목차
1. [환경 준비](#1-환경-준비)
2. [Phase 1: 베이스 모델 테스트](#2-phase-1-베이스-모델-테스트)
3. [Phase 2: Fine-tuning](#3-phase-2-fine-tuning)
4. [다음 단계](#4-다음-단계)

---

## 1. 환경 준비

### 1.1 프로젝트 클론 및 초기 설정

```bash
cd /path/to/mlops-project

# 자동 설정 스크립트 실행
./setup.sh

# 또는 수동 설정:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.2 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# 필수 항목 편집
nano .env
```

**.env 파일 내용:**
```bash
# HuggingFace 토큰 (필수: Gated 모델 사용 시)
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx

# OpenAI API (선택: 합성 데이터 생성 시)
OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# 모델 설정
BASE_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```

### 1.3 GPU 환경 확인

```bash
# GPU 및 환경 확인
python src/check_gpu.py
```

**예상 출력:**
```
============================================================
  MLOps Chatbot Project - GPU Environment Check
============================================================

Python Version: 3.11.14
OS: Linux

✓ CUDA Version: 11.8
✓ PyTorch: 2.1.0
✓ GPU: NVIDIA A100-SXM4-40GB
✓ Total Memory: 40.00 GB
✓ Transformers: 4.35.0
```

---

## 2. Phase 1: 베이스 모델 테스트

### 2.1 기본 모델 로드 및 추론 테스트

```bash
python src/01_test_base_model.py
```

**프롬프트:**
- 모델 로딩 방식 선택 (Full precision / 4-bit quantization)
- 테스트 프롬프트 자동 실행
- 성능 메트릭 출력

**예상 소요 시간:** 5-10분 (모델 다운로드 포함)

### 2.2 Gradio 인터랙티브 데모

```bash
python src/02_gradio_demo.py
```

**브라우저 접속:**
```
http://localhost:7860
```

웹 UI를 통해 모델과 직접 대화할 수 있습니다.

### 2.3 성능 벤치마크 (선택)

```bash
python src/03_benchmark.py
```

다양한 시나리오에서 모델 성능을 측정합니다:
- Latency (응답 시간)
- Throughput (처리량)
- Memory Usage (메모리 사용량)

---

## 3. Phase 2: Fine-tuning

### 3.1 학습 데이터 준비

#### 옵션 A: 공개 데이터셋 사용

```bash
python src/data/01_load_dataset.py
```

**추천 데이터셋:**
1. HuggingFaceH4/no_robots (~10k)
2. tatsu-lab/alpaca (~52k)
3. databricks/databricks-dolly-15k

#### 옵션 B: 합성 데이터 생성

```bash
python src/data/02_generate_synthetic_data.py
```

**설정:**
- 예제 수: 30 (기본값)
- OpenAI API 사용 여부 선택
- MLOps/DevOps 도메인에 특화된 데이터 생성

**출력:** `data/synthetic_train.json`

### 3.2 LoRA Fine-tuning

```bash
python src/train/01_lora_finetune.py
```

**하이퍼파라미터 (기본값):**
```
Epochs: 3
Batch size: 4
Learning rate: 2e-4
LoRA rank: 16
```

**필요 리소스:**
- GPU VRAM: ~14GB (Full precision)
- 학습 시간: ~30-60분 (데이터 크기에 따라)

**출력:**
- 모델: `models/fine-tuned/lora-mistral-custom/`
- MLflow 실험 로그: `mlruns/`

### 3.3 QLoRA Fine-tuning (메모리 효율적)

```bash
python src/train/02_qlora_finetune.py
```

**장점:**
- 4-bit 양자화로 메모리 ~75% 절감
- 필요 VRAM: ~4GB
- 학습 시간은 LoRA와 유사

**출력:**
- 모델: `models/fine-tuned/qlora-mistral-custom/`

### 3.4 MLflow로 실험 확인

```bash
mlflow ui
```

**브라우저 접속:**
```
http://localhost:5000
```

- 여러 실험 비교
- 하이퍼파라미터 추적
- 메트릭 시각화

---

## 4. 다음 단계

### Phase 3: 최적화 (예정)
- vLLM 서빙 구축
- Prompt Engineering
- LangChain 파이프라인

### Phase 4: 프로덕션화 (예정)
- FastAPI 백엔드
- Docker 컨테이너화
- 모니터링 (Prometheus + Grafana)
- CI/CD 파이프라인

---

## 트러블슈팅

### GPU Out of Memory (OOM)
```bash
# 해결 방법 1: 작은 배치 사이즈 사용
batch_size = 2  # 기본값 4 → 2

# 해결 방법 2: QLoRA 사용
python src/train/02_qlora_finetune.py

# 해결 방법 3: Gradient checkpointing
# (스크립트에 이미 포함됨)
```

### HuggingFace 토큰 오류
```bash
# 1. HuggingFace 계정 생성
# 2. Settings → Access Tokens 생성
# 3. .env 파일에 추가
echo "HUGGINGFACE_TOKEN=hf_xxxxx" >> .env
```

### 모델 다운로드 느림
```bash
# 캐시 디렉토리 확인
echo $HF_HOME  # 기본: ~/.cache/huggingface

# 디스크 여유 공간 확인 (최소 50GB 필요)
df -h
```

### MLflow UI 접속 안됨
```bash
# 포트 변경
mlflow ui --port 5001

# 외부 접속 허용
mlflow ui --host 0.0.0.0
```

---

## 유용한 명령어

```bash
# 가상환경 활성화
source venv/bin/activate

# GPU 상태 모니터링
watch -n 1 nvidia-smi

# 학습 로그 실시간 확인
tail -f mlruns/*/meta.yaml

# 디렉토리 크기 확인
du -sh models/

# 프로세스 확인
ps aux | grep python
```

---

## 리소스 요구사항 요약

| 작업 | GPU VRAM | 시간 | 디스크 |
|------|----------|------|--------|
| 모델 다운로드 | - | 5-10분 | ~15GB |
| 추론 테스트 (4-bit) | ~4GB | 1분 | - |
| 추론 테스트 (FP16) | ~14GB | 1분 | - |
| LoRA 학습 | ~14GB | 30-60분 | ~5GB |
| QLoRA 학습 | ~4GB | 30-60분 | ~5GB |

---

## 문의 및 피드백

프로젝트 이슈나 질문은 GitHub Issues에 등록해주세요.

**Happy MLOps! 🚀**
