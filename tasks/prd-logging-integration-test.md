# PRD: 로깅 시스템 통합 테스트

## Introduction/Overview

로깅 개선 작업(feature/logging-audit) 완료 후 전체 스택이 정상 동작하는지 검증한다. vLLM 서비스의 로그 생성, Loki 모니터링 설정의 변경된 로그 경로 참조, 전체 Docker Compose 스택의 통합 테스트를 수행한다.

## Goals

1. vLLM 컨테이너가 logs/vllm/ 디렉토리에 로그 파일을 정상 생성하는지 확인
2. Loki/Alloy 모니터링 설정이 변경된 로그 경로를 올바르게 참조하는지 검증
3. 전체 스택(vLLM, FastAPI, MLflow, Monitoring)이 정상 기동되고 서비스 간 통신이 원활한지 확인
4. 모든 서비스의 로그가 Grafana에서 조회 가능한지 검증

## User Stories

### US-001: vLLM 로그 파일 생성 실제 테스트

**As a** 개발자
**I want** vLLM 컨테이너를 실제로 실행하여 로그 파일이 생성되는지 확인하고 싶다
**So that** 이전 작업(US-001)에서 구현한 로깅 설정이 실제 환경에서 동작하는지 검증할 수 있다

**Acceptance Criteria:**
- docker-compose.serving.yml로 vLLM 컨테이너 기동
- logs/vllm/ 디렉토리에 로그 파일 생성 확인
- 로그 파일에 모델 로딩 메시지 포함 여부 검증
- vLLM API 엔드포인트(/v1/models)에 실제 요청 발송
- 로그 파일에 API 요청 로그 기록 확인
- Typecheck passes

### US-002: Loki/Alloy 모니터링 설정 검증

**As a** DevOps 엔지니어
**I want** Loki/Alloy 설정이 변경된 로그 경로를 올바르게 참조하는지 확인하고 싶다
**So that** 모니터링 시스템이 모든 서비스 로그를 수집할 수 있다

**Acceptance Criteria:**
- deployment/monitoring/alloy/ 또는 promtail 설정 파일 확인
- 로그 경로 설정이 logs/vllm/, logs/fastapi/, logs/mlflow/를 참조하는지 검증
- 경로가 잘못되었거나 누락된 경우 수정
- docker-compose.monitoring.yml로 모니터링 스택 기동
- Loki API(/loki/api/v1/labels)로 수집 중인 로그 라벨 확인
- Typecheck passes

### US-003: Grafana 로그 조회 테스트

**As a** 개발자
**I want** Grafana에서 모든 서비스의 로그를 조회할 수 있는지 확인하고 싶다
**So that** 실시간 로그 모니터링이 가능한지 검증할 수 있다

**Acceptance Criteria:**
- Grafana UI(http://localhost:3000)에 접근
- Loki 데이터소스 연결 상태 확인
- Explore 메뉴에서 {service="vllm"} 쿼리 실행
- Explore 메뉴에서 {service="fastapi"} 쿼리 실행
- Explore 메뉴에서 {service="mlflow"} 쿼리 실행
- 각 서비스의 로그가 조회되면 성공, 조회되지 않으면 Alloy 설정 수정
- Typecheck passes

### US-004: 전체 스택 통합 테스트

**As a** 시스템 관리자
**I want** 전체 Docker Compose 스택을 기동하여 모든 서비스가 정상 동작하는지 확인하고 싶다
**So that** 프로덕션 환경에 배포할 준비가 되었는지 검증할 수 있다

**Acceptance Criteria:**
- docker-compose.yml(전체 스택) 또는 개별 스택 파일로 모든 서비스 기동
- 각 서비스 헬스체크 엔드포인트 확인 (vLLM, FastAPI, MLflow, Prometheus, Grafana, Loki)
- FastAPI → vLLM 추론 요청 테스트 (curl 또는 pytest)
- MLflow UI 접근 확인 (http://localhost:5050)
- scripts/check_logs.sh 실행하여 모든 서비스 로그 검증
- 모든 서비스가 정상 동작하고 로그가 생성되면 성공
- Typecheck passes

### US-005: 통합 테스트 결과 문서화

**As a** 개발자
**I want** 통합 테스트 결과를 문서화하고 싶다
**So that** 향후 배포 시 참고할 수 있다

**Acceptance Criteria:**
- 테스트 결과를 .context/terminal/integration_test_$(date +%s).log에 기록
- 각 서비스별 테스트 통과/실패 상태 요약
- 발견된 문제와 해결 방법 기록
- docs/references/LOGGING.md에 통합 테스트 섹션 추가 (선택)
- deployment/CLAUDE.md에 전체 스택 기동 방법 업데이트 (필요 시)
- Typecheck passes

## Functional Requirements

**FR-1**: vLLM 컨테이너는 logs/vllm/ 디렉토리에 타임스탬프 기반 로그 파일을 생성해야 한다
**FR-2**: Loki는 모든 서비스(vllm, fastapi, mlflow) 로그를 수집해야 한다
**FR-3**: Grafana에서 각 서비스별로 로그를 필터링하여 조회할 수 있어야 한다
**FR-4**: 전체 스택 기동 시 모든 서비스가 헬스체크를 통과해야 한다
**FR-5**: FastAPI와 vLLM 간 통신이 정상 동작하고 로그가 기록되어야 한다

## Non-Goals

- 새로운 로깅 기능 추가 (기존 기능 검증만)
- 성능 테스트 또는 부하 테스트
- 프로덕션 환경 배포
- 로그 보관 정책 설정

## Technical Considerations

- vLLM 컨테이너 실행 시 모델 다운로드에 시간이 소요될 수 있음
- Loki 설정은 deployment/monitoring/alloy/ 또는 promtail 디렉토리에 위치
- Docker Compose 스택은 docker/ 디렉토리에 분리되어 있음 (mlflow, serving, monitoring)
- 로그 검증 스크립트는 scripts/check_logs.sh 사용

## Success Metrics

- [ ] vLLM 로그 파일 생성 및 API 요청 로그 기록
- [ ] Loki가 모든 서비스 로그를 수집
- [ ] Grafana에서 각 서비스 로그 조회 가능
- [ ] 전체 스택 헬스체크 통과
- [ ] FastAPI → vLLM 추론 테스트 성공
- [ ] scripts/check_logs.sh 검증 통과

## Open Questions

- vLLM 컨테이너 실행 시 어떤 모델을 사용할 것인가? (기본 모델 또는 fine-tuned 모델)
- Grafana 대시보드는 기존 것을 사용하는가, 새로 생성하는가?
- 통합 테스트 자동화 스크립트 작성이 필요한가?
