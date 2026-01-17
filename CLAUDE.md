# MLOps Chatbot Project

LLM Fine-tuning → 프로덕션 배포 MLOps 파이프라인

---

## ⚠️ 필수 실행 규칙 (모든 작업 전 확인)

### 1. 터미널 로그 기록
모든 Bash 명령어(빌드, 테스트, 설치, 실행) 실행 시 `.context/terminal/`에 로그 저장:
```bash
[명령어] 2>&1 | tee .context/terminal/[명령어]_$(date +%s).log
```

### 2. 서브 CLAUDE.md 관리
- 새 디렉토리/모듈 생성 시 → 해당 디렉토리에 서브 CLAUDE.md 생성
- 기존 구조 변경 시 → 관련 서브 CLAUDE.md 업데이트

### 3. Git 커밋 규칙
- 커밋 메시지는 **한글**로 작성
- `Co-Authored-By` 태그 **사용 금지**
- 형식: `<type>: <한글 설명>` (feat, fix, docs, refactor, test, chore)

### 4. 작업 완료 체크리스트
- [ ] 터미널 로그 저장했는가?
- [ ] 서브 CLAUDE.md 업데이트 필요한가?
- [ ] 세션 히스토리 기록했는가?

---

## 현재 상태

- **Phase**: 2 (Fine-tuning 완료)
- **베이스 모델**: LLaMA-3-8B-Instruct
- **GPU**: RTX 5090 (31GB) + RTX 5060 Ti (15GB)
- **배포된 모델**: [2shlee/llama3-8b-ko-chat-v1](https://huggingface.co/2shlee/llama3-8b-ko-chat-v1)
- **리팩토링**: 클린 아키텍처 적용 완료 → `src/serve/CLAUDE.md`

## 기술 스택

| 분류 | 기술 |
|------|------|
| Core ML | PyTorch 2.1+, Transformers 4.35+, PEFT, bitsandbytes |
| Serving | vLLM, FastAPI, Gradio |
| MLOps | MLflow, DVC, LangChain |
| Monitoring | Prometheus, Grafana, Loki, structlog |
| DevOps | Docker, Docker Compose |
| Database | SQLAlchemy 2.0+, Alembic (마이그레이션), SQLite |
| Config | pydantic-settings |

## 서브 CLAUDE.md 목록

| 경로 | 설명 |
|------|------|
| [src/serve/CLAUDE.md](src/serve/CLAUDE.md) | FastAPI 서빙 (클린 아키텍처) |
| [src/serve/migrations/CLAUDE.md](src/serve/migrations/CLAUDE.md) | Alembic 마이그레이션 |
| [src/train/CLAUDE.md](src/train/CLAUDE.md) | LoRA/QLoRA Fine-tuning |
| [src/data/CLAUDE.md](src/data/CLAUDE.md) | 데이터 파이프라인 |
| [src/evaluate/CLAUDE.md](src/evaluate/CLAUDE.md) | 모델 평가 |
| [src/utils/CLAUDE.md](src/utils/CLAUDE.md) | 로깅 유틸리티 |
| [deployment/CLAUDE.md](deployment/CLAUDE.md) | Docker 배포 |

## 디렉토리 구조

```
src/
├── train/       → src/train/CLAUDE.md
├── serve/       → src/serve/CLAUDE.md (클린 아키텍처 적용)
│   ├── main.py              # FastAPI 엔트리포인트
│   ├── database.py          # SQLAlchemy 설정
│   ├── migrations/          # Alembic 마이그레이션
│   ├── core/                # 설정, LLM 클라이언트
│   ├── models/              # ORM 모델
│   ├── schemas/             # Pydantic 스키마
│   ├── cruds/               # DB CRUD 함수
│   └── routers/             # API 라우터
├── data/        → src/data/CLAUDE.md
├── evaluate/    → src/evaluate/CLAUDE.md
└── utils/       → src/utils/CLAUDE.md
deployment/      → deployment/CLAUDE.md
tests/serve/          # API 테스트
docs/
├── guides/           # Git 서브모듈 (CLAUDE-DOCS 저장소)
├── references/       # 참조 가이드 (LOGGING.md, VLLM.md)
└── plans/            # 리팩토링 계획 문서
models/
├── base/             # HuggingFace 캐시
├── fine-tuned/       # LoRA 어댑터 저장
└── downloaded/       # HF Hub에서 다운로드한 모델
data/
├── processed/        # JSONL 형식 (no_robots: 9,499건)
└── synthetic_train.json  # MLOps/DevOps 특화 합성 데이터
results/              # 실험 결과
mlruns/               # MLflow 실험 저장소
logs/                 # 구조화된 로그 (JSON)
```

## 주요 명령어

```bash
# pyenv 가상환경 (자동 활성화 - .python-version)
cd /path/to/mlops-project  # mlops-project 환경 자동 적용

# GPU 및 환경 확인
python src/check_gpu.py

# 학습
python src/train/01_lora_finetune.py
python src/train/02_qlora_finetune.py
mlflow ui --port 5000

# 서빙
python src/serve/01_vllm_server.py   # vLLM :8000
python -m src.serve.main             # FastAPI :8080 (클린 아키텍처)

# 테스트
python -m pytest tests/serve/ -v

# DB 마이그레이션 (Alembic) - 프로젝트 루트에서 실행
alembic current                        # 현재 상태
alembic revision --autogenerate -m "설명"  # 마이그레이션 생성
alembic upgrade head                   # 적용

# Docker (전체 스택)
docker-compose up -d
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `HUGGINGFACE_TOKEN` | - | Gated 모델 접근 (필수) |
| `VLLM_BASE_URL` | http://localhost:8000/v1 | vLLM 서버 |
| `FASTAPI_PORT` | 8080 | FastAPI 포트 |
| `DATABASE_URL` | sqlite+aiosqlite:///./mlops_chat.db | DB |
| `ENABLE_AUTH` | false | 인증 활성화 |

환경 파일: `cp env.example .env`

## 세션 관리 규칙

### 세션 시작 시
1. `.context/history/`에서 최근 세션 파일 확인
2. 이전 세션의 TODO 파악
3. 중단된 작업이 있으면 이어서 진행

### 세션 종료 시
1. `.context/history/session_YYYY-MM-DD_HH-MM.md` 파일 생성
2. 다음 내용 기록:
   - 완료한 작업
   - 주요 결정사항
   - 다음 세션 TODO

## 참고 문서

- [CLAUDE.md 가이드라인](docs/guides/CLAUDE.md) - 문서 작성 규칙 (서브모듈)
- [로깅 가이드](docs/references/LOGGING.md) - 구조화된 로깅
- [vLLM 가이드](docs/references/VLLM.md) - vLLM 서빙
