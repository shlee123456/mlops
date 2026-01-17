# MLOps Chatbot Project

LLM Fine-tuning → 프로덕션 배포 MLOps 파이프라인

## 현재 상태

<<<<<<< HEAD
- **Phase**: 2 (Fine-tuning 완료)
- **베이스 모델**: LLaMA-3-8B-Instruct
- **GPU**: RTX 5090 (31GB) + RTX 5060 Ti (15GB)
- **배포된 모델**: [2shlee/llama3-8b-ko-chat-v1](https://huggingface.co/2shlee/llama3-8b-ko-chat-v1)
- **리팩토링**: 클린 아키텍처 적용 완료 → `src/serve/CLAUDE.md`
=======
## ⚠️ 필수 실행 규칙 (모든 작업 전 확인)

> **이 규칙은 모든 작업에서 반드시 준수해야 합니다.**

### 1. 터미널 로그 기록
모든 Bash 명령어(빌드, 테스트, 설치, 실행) 실행 시 `.context/terminal/`에 로그 저장:
```bash
[명령어] 2>&1 | tee .context/terminal/[명령어]_$(date +%s).log
```

### 2. 서브 CLAUDE.md 관리
- 새 디렉토리/모듈 생성 시 → 해당 디렉토리에 서브 CLAUDE.md 생성
- 기존 구조 변경 시 → 관련 서브 CLAUDE.md 업데이트
- 서브 CLAUDE.md는 반드시 루트 CLAUDE.md 참조

### 3. 세션 관리
- **세션 시작**: `.context/history/`에서 최근 기록 확인, 이전 TODO 파악
- **세션 종료**: `.context/history/session_YYYY-MM-DD_HH-MM.md`에 작업 내용 기록

### 4. Git 커밋 규칙
- 커밋 메시지는 **한글**로 작성
- `Co-Authored-By` 태그 **사용 금지** (Claude 공동 작성자 표기 안 함)
- 커밋 메시지 형식:
```
<type>: <한글 설명>

<본문 (선택)>
```
- type: feat, fix, docs, refactor, test, chore 등

### 5. 작업 완료 체크리스트
- [ ] 터미널 로그 저장했는가?
- [ ] 서브 CLAUDE.md 업데이트 필요한가?
- [ ] 세션 히스토리 기록했는가?
- [ ] Git 커밋 시 한글 메시지 사용했는가?

---

## 목표

- **루트 CLAUDE.md**가 모든 서브 CLAUDE.md를 통제
- **서브 CLAUDE.md**가 루트 CLAUDE.md를 참조
- **세션 간 연속성** 확보: 새 세션에서도 작업 흐름이 매끄럽게 이어짐

---

## 최종 디렉토리 구조

```
project-root/
├── CLAUDE.md                          # 루트: 전역 규칙, 서브 문서 목록
├── .context/                          # 맥락 관리 (gitignore 대상)
│   ├── history/
│   │   └── session_YYYY-MM-DD_HH-MM.md
│   └── terminal/
│       └── build_1705483200.log
├── src/
│   └── CLAUDE.md                      # 서브: src 관련 규칙 + 루트 참조
├── api/
│   └── CLAUDE.md                      # 서브: api 관련 규칙 + 루트 참조
└── .gitignore                         # .context/ 제외
```

---

## 토큰 관리 가이드라인

계층형 CLAUDE.md의 토큰 절감 효과를 극대화하기 위한 권장 사항입니다.

### 토큰 절감 원리

```
# 단일 CLAUDE.md (비효율)
모든 규칙이 매번 로드됨 → 토큰 낭비

# 계층형 CLAUDE.md (효율적)
src/ 작업 시: 루트 + src/CLAUDE.md만 로드
api/ 작업 시: 루트 + api/CLAUDE.md만 로드
→ 불필요한 컨텍스트 제외
```

### 문서 크기 제한

| 문서 유형 | 권장 크기 | 최대 크기 |
|----------|----------|----------|
| 루트 CLAUDE.md | 300줄 이하 | 500줄 |
| 서브 CLAUDE.md | 100줄 이하 | 200줄 |
| 세션 히스토리 | 50줄 이하 | 100줄 |

### 루트 CLAUDE.md 최적화

```markdown
## ✅ 포함할 것
- 프로젝트 개요 (2-3문장)
- 기술 스택 (목록만)
- 전역 규칙 (핵심만)
- 서브 문서 경로 목록

## ❌ 제외할 것
- 상세한 코드 예시 → 서브 문서로
- 특정 디렉토리 규칙 → 해당 서브 문서로
- 긴 설명 → "자세한 내용은 [경로] 참조"
```

### 서브 CLAUDE.md 최적화

```markdown
## ✅ 올바른 예시
> **상위 문서**: [루트 CLAUDE.md](../CLAUDE.md) 참조
## 로컬 규칙
- 이 디렉토리 특화 규칙만 작성

## ❌ 잘못된 예시 (중복)
## 기술 스택
- React, TypeScript... (루트와 중복!)
## 전역 규칙
- 네이밍 컨벤션... (루트와 중복!)
```

### 히스토리 관리

```
세션 히스토리 정리 규칙을 루트 CLAUDE.md에 추가해줘:
- .context/history/에 최근 5개 세션만 유지
- 7일 이상 된 히스토리는 .context/archive/로 이동
- 30일 이상 된 아카이브는 삭제
```

**자동 정리 스크립트 (선택사항):**
```bash
# 7일 이상 된 히스토리 아카이브
find .context/history -name "*.md" -mtime +7 -exec mv {} .context/archive/ \;

# 30일 이상 된 아카이브 삭제
find .context/archive -name "*.md" -mtime +30 -delete
```

### 히스토리 고도화 (프로젝트 규모별)

프로젝트 규모에 따라 히스토리 관리 전략을 선택합니다.

#### Level 1: 날짜별 관리 (소규모 프로젝트)

현재 기본 방식입니다.
```
.context/history/
├── session_2025-01-15_09-30.md
├── session_2025-01-16_14-00.md
└── session_2025-01-17_10-15.md
```

#### Level 2: 카테고리 분류 (중규모 프로젝트)

```
히스토리를 카테고리별로 분류하는 구조로 변경해줘:

.context/history/
├── features/          # 기능 개발 관련
│   ├── auth_login.md
│   └── payment.md
├── bugs/              # 버그 수정 관련
│   ├── api_timeout.md
│   └── memory_leak.md
├── decisions/         # 주요 결정사항
│   ├── tech_stack.md
│   └── architecture.md
└── sessions/          # 일반 세션 (분류 전)
    └── session_2025-01-17.md
```

**세션 종료 프롬프트:**
```
세션을 종료하고 히스토리를 저장해줘:
1. 작업 유형 판단 (feature/bug/decision)
2. 적절한 카테고리 폴더에 저장
3. 기존 관련 파일이 있으면 내용 추가
```

#### Level 3: 인덱스 파일 도입 (대규모/장기 프로젝트)

```
히스토리 인덱스 시스템을 구축해줘:

.context/history/
├── index.md           # 검색용 인덱스 (핵심!)
├── features/
├── bugs/
├── decisions/
└── sessions/
```

**index.md 템플릿:**
```markdown
# 히스토리 인덱스

## 최근 작업 (최근 5개)
| 날짜 | 주제 | 파일 | 태그 |
|------|------|------|------|
| 2025-01-17 | API 리팩토링 | features/api_refactor.md | #api #refactor |
| 2025-01-16 | 로그인 버그 수정 | bugs/auth_error.md | #auth #bugfix |

## 태그 인덱스
- #auth: features/auth_login.md, bugs/auth_error.md
- #api: features/api_refactor.md, bugs/api_timeout.md
- #architecture: decisions/architecture.md

## 미완료 TODO (통합)
- [ ] 결제 연동 테스트 → features/payment.md
- [ ] API 문서화 → features/api_refactor.md
```

**세션 시작 프롬프트:**
```
.context/history/index.md를 확인하고:
1. 미완료 TODO 목록 파악
2. 오늘 작업할 내용과 관련된 태그 검색
3. 관련 히스토리 파일 요약해줘
```

**태그 검색 프롬프트:**
```
.context/history/index.md에서 #auth 태그가 붙은 파일들을 찾아서 내용을 요약해줘.
```

**세션 종료 프롬프트:**
```
세션을 종료하고:
1. 적절한 카테고리에 히스토리 저장
2. 태그 추출해서 index.md 업데이트
3. 미완료 TODO를 index.md에 통합
```

#### 프로젝트 규모별 권장 사항

| 프로젝트 규모 | 권장 Level | 이유 |
|--------------|-----------|------|
| 소규모 (1-2주) | Level 1 | 관리 오버헤드 최소화 |
| 중규모 (1-3개월) | Level 2 | 카테고리로 빠른 탐색 |
| 대규모 (3개월+) | Level 3 | 인덱스로 효율적 검색 |

#### 히스토리 고도화 마이그레이션

기존 날짜별 히스토리를 카테고리 구조로 전환:
```
기존 .context/history/의 세션 파일들을 분석해서:
1. 각 세션의 주요 주제 파악
2. 적절한 카테고리로 분류 (features/bugs/decisions)
3. 관련 세션들을 하나의 주제 파일로 통합
4. index.md 생성
```

### 터미널 로그 관리

```
터미널 로그 정리 규칙을 루트 CLAUDE.md에 추가해줘:
- .context/terminal/에 최근 10개 로그만 유지
- 오래된 로그는 자동 삭제
```

**자동 정리 스크립트 (선택사항):**
```bash
# 최근 10개 제외하고 삭제
ls -t .context/terminal/*.log | tail -n +11 | xargs rm -f
```

### 토큰 사용량 모니터링

```
현재 CLAUDE.md 구조의 토큰 사용량을 분석해줘:
1. 루트 CLAUDE.md 예상 토큰 수
2. 각 서브 CLAUDE.md 예상 토큰 수
3. 최적화 필요한 부분 제안
```

### 토큰 절감 체크리스트

프로젝트 주기적으로 점검:

- [ ] 루트 CLAUDE.md가 500줄 이하인가?
- [ ] 서브 CLAUDE.md가 200줄 이하인가?
- [ ] 루트와 서브 간 내용 중복이 없는가?
- [ ] 히스토리 파일이 5개 이하인가?
- [ ] 터미널 로그가 10개 이하인가?
- [ ] 불필요한 서브 CLAUDE.md가 없는가?

---

## 1단계: 환경 설정

맥락 관리를 위한 디렉토리를 생성하고 gitignore에 추가합니다.

```
다음 환경 설정을 진행해줘:
1. .context/history/ 디렉토리 생성
2. .context/terminal/ 디렉토리 생성
3. .gitignore에 .context/ 추가 (개인 로컬 기록이므로)
```

---

## 2단계: 루트 CLAUDE.md 생성

프로젝트 전체를 통제하는 루트 CLAUDE.md를 생성합니다.

```
프로젝트를 분석하고 루트 CLAUDE.md를 생성해줘.

다음 구조로 작성해줘:
1. 프로젝트 개요
2. 기술 스택
3. 전역 코딩 컨벤션
4. 서브 CLAUDE.md 목록 (각 디렉토리별 문서 경로)
5. 세션 관리 규칙
   - 세션 시작 시: .context/history/에서 최근 세션 파일 확인
   - 세션 종료 시: 주요 결정사항과 진행상황 기록
6. 자주 사용하는 명령어
```

**루트 CLAUDE.md 템플릿:**

```markdown
# 프로젝트명

## 개요
[프로젝트 설명]
>>>>>>> 28c0dc8 (docs: CLAUDE.md에 Git 커밋 규칙 추가)

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

## 디렉토리 구조

```
src/
├── train/       → src/train/CLAUDE.md
├── serve/       → src/serve/CLAUDE.md (클린 아키텍처 적용)
│   ├── main.py              # FastAPI 엔트리포인트
│   ├── database.py          # SQLAlchemy 설정
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
├── guides/           # 참조 가이드 (LOGGING.md, VLLM.md)
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

## 환경변수 (전체)

```bash
# 필수
HUGGINGFACE_TOKEN        # Gated 모델 접근

# MLflow
MLFLOW_TRACKING_URI      # MLflow 서버 (기본: ./mlruns)

# 서빙
VLLM_ENDPOINT            # vLLM 서버 (기본: http://localhost:8000)
GPU_MEMORY_UTILIZATION   # GPU 메모리 사용률 (기본: 0.9)
MAX_MODEL_LEN            # 최대 시퀀스 (기본: 4096)
MODEL_PATH               # 모델 경로
API_KEY                  # API 인증 키 (기본: your-secret-api-key)
ENABLE_AUTH              # 인증 활성화 (기본: false)

# 데이터베이스
DATABASE_URL             # DB 연결 (기본: sqlite:///./data/chat.db)

# 로깅
LOG_DIR                  # 로그 디렉토리 (기본: ./logs)
LOG_LEVEL                # 로그 레벨 (기본: INFO)
```

## 코딩 컨벤션

| 항목 | 규칙 | 예시 |
|------|------|------|
| Python | 3.10+, Black, isort | - |
| 파일명 | `{순번}_{기능}.py` | `01_lora_finetune.py` |
| 함수 | snake_case + type hints | `def load_data(path: str) -> Dataset:` |
| 클래스 | PascalCase | `TrainingLogger` |
| 상수 | UPPER_SNAKE | `LogType.TRAINING` |
| Docstring | Google 스타일 | `Args:`, `Returns:` |
| API 모델 | Pydantic + Field | `Field(..., description="설명")` |

## 주요 명령어

```bash
source venv/bin/activate          # 가상환경

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

## 커밋 규칙

`.claude/skills.md` 참조 - 한글 커밋 메시지 사용

```
추가: 새로운 기능
수정: 기존 기능 변경
버그수정: 버그 해결
문서: 문서 작성/수정
리팩토링: 코드 구조 개선
```

## 주의사항

1. **OOM 방지**: `batch_size`, `max_length` 조절
2. **MLflow**: 학습 전 `mlflow.set_experiment()` 호출
3. **로깅**: `src/utils/logging_utils.py` 구조화 로거 사용
4. **HF 토큰**: Gated 모델 접근 시 환경변수 필수
