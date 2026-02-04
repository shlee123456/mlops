# PRD: 로깅 시스템 점검 및 개선

## 1. Introduction/Overview

MLOps 프로젝트의 모든 서비스가 `logs/` 디렉토리에 파일 로그를 정상적으로 생성하는지 점검하고, 로그가 생성되지 않는 서비스의 설정을 개선합니다.

**현재 상황:**
- `logs/` 디렉토리에 6개 서브디렉토리 존재: `fastapi`, `inference`, `mlflow`, `system`, `training`, `vllm`
- Docker 컨테이너는 실행 중이나 일부 서비스에서 파일 로그가 생성되지 않을 가능성

**해결할 문제:**
- 각 서비스의 로깅 설정이 올바르게 구성되어 있는지 불명확
- 파일 로그 생성 여부를 체계적으로 검증하지 않음
- 로그가 누락되면 디버깅 및 모니터링 불가

## 2. Goals

1. 모든 서비스(vLLM, FastAPI, MLflow, Inference, System, Training)의 로깅 설정 검증
2. 각 서비스가 `logs/[service]/` 디렉토리에 파일 로그를 생성하는지 확인
3. 로그가 생성되지 않는 서비스의 설정을 수정하여 로그 생성 활성화
4. 로깅 설정 문서화 (각 서비스별 CLAUDE.md 업데이트)

## 3. User Stories

### US-001: vLLM 서비스 로그 파일 생성 점검 및 개선
**설명:** vLLM 컨테이너가 `logs/vllm/` 디렉토리에 파일 로그를 생성하는지 확인하고, 생성되지 않으면 설정 수정

**수용 기준:**
- `deployment/serving/Dockerfile.vllm`의 로깅 설정 확인
- vLLM 시작 스크립트가 로그를 파일로 리다이렉트하는지 검증
- 컨테이너 실행 후 `logs/vllm/` 디렉토리에 로그 파일 생성 확인
- 로그 파일에 vLLM 시작 메시지, 모델 로딩 로그 포함 여부 확인
- 로그가 생성되지 않으면 스크립트 수정
- 수정 후 테스트하여 로그 생성 재확인

### US-002: FastAPI 서비스 로그 생성 검증
**설명:** FastAPI 서비스가 `logs/fastapi/` 디렉토리에 구조화된 JSON 로그를 생성하는지 확인

**수용 기준:**
- `src/serve/main.py`에서 로깅 설정 확인 (logging_utils 사용)
- `logs/fastapi/` 디렉토리에 JSON 로그 파일 존재 확인
- API 요청 발생 시 로그가 실시간으로 기록되는지 테스트 (curl 또는 pytest)
- 로그에 request_id, method, path, status_code, duration_ms 포함 여부 검증
- 문제 발견 시 `src/utils/logging_utils.py` 설정 수정

### US-003: MLflow 서비스 로그 생성 검증
**설명:** MLflow 컨테이너가 `logs/mlflow/` 디렉토리에 로그를 생성하는지 확인

**수용 기준:**
- `docker/docker-compose.mlflow.yml`의 볼륨 마운트 확인 (`logs/mlflow` 매핑)
- MLflow 컨테이너 실행 후 `logs/mlflow/` 디렉토리에 로그 파일 생성 확인
- 로그 파일에 MLflow 서버 시작 메시지, 요청 로그 포함 여부 검증
- 로그가 생성되지 않으면 Dockerfile 또는 entrypoint 스크립트 수정
- 수정 후 재시작하여 로그 생성 재확인

### US-004: Inference/System/Training 로그 디렉토리 사용 여부 점검
**설명:** `logs/inference`, `logs/system`, `logs/training` 디렉토리를 사용하는 코드가 있는지 확인하고, 로그 생성 검증

**수용 기준:**
- `src/` 디렉토리에서 `logs/inference`, `logs/system`, `logs/training` 경로를 사용하는 코드 검색
- 각 로그 타입을 사용하는 스크립트/모듈 식별
- 해당 스크립트 실행 후 로그 파일 생성 확인
- 로그가 생성되지 않으면 `src/utils/logging_utils.py` 설정 검토
- 사용하지 않는 디렉토리는 문서에 명시 (미사용 또는 미래 확장용)

### US-005: 로깅 설정 문서화 및 테스트 작성
**설명:** 각 서비스의 로깅 설정을 문서화하고, 로그 생성을 검증하는 간단한 테스트 작성

**수용 기준:**
- `docs/references/LOGGING.md`에 각 서비스별 로그 경로, 포맷, 설정 방법 업데이트
- `src/serve/CLAUDE.md`, `deployment/CLAUDE.md`에 로깅 설정 정보 추가
- 로그 생성을 검증하는 간단한 Bash 스크립트 작성 (`scripts/check_logs.sh`)
- 스크립트 실행 시 각 서비스의 로그 파일 존재 여부 출력
- 터미널 로그 기록: `.context/terminal/logging_audit_$(date +%s).log`

## 4. Functional Requirements

**FR-1:** 모든 Docker 서비스는 `logs/[service]` 디렉토리에 파일 로그를 생성해야 함
**FR-2:** 로그 파일은 JSON 형식(구조화) 또는 일반 텍스트 형식으로 저장
**FR-3:** 로그 파일에는 타임스탬프, 로그 레벨, 메시지가 포함되어야 함
**FR-4:** Docker 컨테이너 재시작 후에도 로그가 지속적으로 누적되어야 함
**FR-5:** 로그 디렉토리가 존재하지 않으면 자동 생성되어야 함

## 5. Non-Goals

- Loki/Grafana 연동 설정은 이번 작업 범위 밖
- 로그 로테이션 정책 설정은 제외 (기본 Docker 로그 설정 사용)
- 로그 포맷 통일은 제외 (각 서비스의 기존 포맷 유지)

## 6. Technical Considerations

- Docker Compose 볼륨 마운트: `../logs/[service]:/logs`
- Python 로깅: `src/utils/logging_utils.py` (structlog 기반)
- vLLM 로깅: stdout을 파일로 리다이렉트 필요 (`tee` 또는 `>> /logs/vllm.log`)
- MLflow 로깅: gunicorn/Flask 로그 설정 확인

## 7. Success Metrics

1. **로그 파일 생성률**: 6개 서비스 중 몇 개가 파일 로그를 생성하는가?
   - 목표: 100% (6/6)
2. **로그 내용 완전성**: 로그 파일에 필수 정보(시간, 레벨, 메시지) 포함 여부
   - 목표: 모든 서비스에서 필수 정보 포함
3. **문서화**: 로깅 설정이 CLAUDE.md 및 LOGGING.md에 문서화되었는가?
   - 목표: 문서 업데이트 완료

## 8. Open Questions

- Q1: inference, system, training 로그 디렉토리는 실제로 사용 중인가, 아니면 미래 확장용인가?
- Q2: vLLM 로그를 파일로 쓸 때 stdout과 stderr를 분리할 것인가?
- Q3: 로그 보존 기간 정책이 필요한가?

## 9. Implementation Notes

- **우선순위**: vLLM > FastAPI > MLflow > 나머지
- **테스트 방법**:
  - 컨테이너 재시작 → 5초 대기 → `ls -la logs/[service]/` 확인
  - 로그 파일이 있으면 `tail -20` 으로 내용 확인
- **롤백 계획**: 수정 전 Dockerfile/스크립트 백업 (Git commit)
