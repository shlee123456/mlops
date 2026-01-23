# Docker Compose 분리 및 디렉토리 구조 개편

> **작성일**: 2026-01-21
> **상태**: 검토 중

## 개요

기능별로 docker-compose 파일을 분리하고, deployment 디렉토리를 스택별로 재구성하여 선택적 실행, 독립 배포, 유지보수성을 개선합니다.

## TODO

- [ ] docker/ 디렉토리 생성 및 분리된 compose 파일 작성
- [ ] deployment/ 디렉토리를 스택별로 재구성 (mlflow, serving, monitoring)
- [ ] Dockerfile 경로 및 context 업데이트
- [ ] 각 스택 개별 실행 테스트
- [ ] README.md, CLAUDE.md, deployment/CLAUDE.md 문서 업데이트

---

## 현재 구조 분석

현재 단일 `docker-compose.yml` (271줄)에 4개 스택이 혼재:

```
docker-compose.yml
├── MLflow Stack: postgres, minio, mlflow-server
├── Serving Stack: vllm-server, fastapi-server
└── Monitoring Stack: loki, alloy, prometheus, grafana
```

**문제점**:
- 전체 스택을 한 번에 올려야 함 (개발 시 불편)
- 파일이 커서 수정 시 실수 가능성 증가
- 스택별 독립 배포 어려움

---

## 제안 구조

### 1. Docker Compose 파일 분리

```
docker/
├── docker-compose.yml          # 공통 네트워크/볼륨 정의 + 전체 실행용
├── docker-compose.mlflow.yml   # MLflow Stack (postgres, minio, mlflow)
├── docker-compose.serving.yml  # Serving Stack (vllm, fastapi)
├── docker-compose.monitoring.yml # Monitoring Stack (prometheus, grafana, loki, alloy)
└── .env.example                # Docker 전용 환경변수 템플릿
```

**실행 방법**:
```bash
# 개별 스택 실행
docker compose -f docker/docker-compose.mlflow.yml up -d
docker compose -f docker/docker-compose.serving.yml up -d
docker compose -f docker/docker-compose.monitoring.yml up -d

# 전체 스택 실행 (기존과 동일)
docker compose -f docker/docker-compose.yml up -d

# 또는 여러 파일 조합
docker compose -f docker/docker-compose.mlflow.yml -f docker/docker-compose.serving.yml up -d
```

### 2. Deployment 디렉토리 재구성

```
deployment/
├── CLAUDE.md
├── README.md
│
├── mlflow/                     # MLflow Stack
│   ├── Dockerfile
│   └── configs/
│       └── (필요시 추가)
│
├── serving/                    # Model Serving Stack
│   ├── Dockerfile.vllm
│   └── Dockerfile.fastapi
│
└── monitoring/                 # Monitoring Stack
    └── configs/
        ├── alloy/
        │   └── config.alloy
        ├── grafana/
        │   ├── dashboards/
        │   └── provisioning/
        ├── loki/
        │   └── loki-config.yaml
        └── prometheus/
            └── prometheus.yml
```

### 3. 전체 프로젝트 구조 (변경 후)

```
mlops-project/
├── docker/                     # [NEW] Docker Compose 파일들
│   ├── docker-compose.yml
│   ├── docker-compose.mlflow.yml
│   ├── docker-compose.serving.yml
│   ├── docker-compose.monitoring.yml
│   └── .env.example
│
├── deployment/                 # [RESTRUCTURE] 스택별 Dockerfile/Config
│   ├── mlflow/
│   ├── serving/
│   └── monitoring/
│
├── src/                        # [유지] 소스 코드
├── data/                       # [유지] 데이터
├── models/                     # [유지] 모델
├── logs/                       # [유지] 로그
├── tests/                      # [유지] 테스트
└── docs/                       # [유지] 문서
```

---

## 스택별 docker-compose 구성

### docker-compose.mlflow.yml
```yaml
services:
  postgres:     # 메타데이터 저장
  minio:        # 아티팩트 저장 (S3 호환)
  mlflow-server: # MLflow UI/API

networks:
  mlops-network:
    external: true  # 공유 네트워크

volumes:
  postgres_data:
  minio_data:
```

### docker-compose.serving.yml
```yaml
services:
  vllm-server:    # GPU 모델 서빙
  fastapi-server: # API Gateway

networks:
  mlops-network:
    external: true

volumes:
  fastapi_data:
```

### docker-compose.monitoring.yml
```yaml
services:
  loki:       # 로그 저장
  alloy:      # 메트릭/로그 수집
  prometheus: # 메트릭 저장
  grafana:    # 시각화

networks:
  mlops-network:
    external: true

volumes:
  loki_data:
  prometheus_data:
  grafana_data:
  alloy_data:
```

### docker-compose.yml (전체 스택)
```yaml
# 공통 네트워크 정의
networks:
  mlops-network:
    driver: bridge

# include로 모든 스택 포함 (Docker Compose v2.20+)
include:
  - docker-compose.mlflow.yml
  - docker-compose.serving.yml
  - docker-compose.monitoring.yml
```

---

## 장점

| 항목 | 개선 효과 |
|------|-----------|
| **개발 편의성** | 필요한 스택만 선택적 실행 (예: serving만 올리고 테스트) |
| **배포 유연성** | 스택별 독립 배포/스케일링 가능 |
| **유지보수성** | 파일당 ~80줄로 관리 용이, 변경 영향 범위 축소 |
| **가독성** | 스택별 관심사 분리, 설정 찾기 쉬움 |

---

## 마이그레이션 전략

1. **Phase 1**: `docker/` 디렉토리 생성 및 compose 파일 분리
2. **Phase 2**: `deployment/` 디렉토리 재구성 (스택별)
3. **Phase 3**: 기존 `docker-compose.yml` deprecated 처리 후 삭제
4. **Phase 4**: 문서 업데이트 (README, CLAUDE.md)

---

## 고려사항

- **네트워크 공유**: 모든 스택이 `mlops-network`를 공유해야 서비스 간 통신 가능
- **의존성**: serving → mlflow (선택적), monitoring → serving/mlflow (메트릭 수집)
- **Docker Compose 버전**: `include` 구문은 v2.20+ 필요 (없으면 `-f` 플래그 사용)
- **기존 볼륨 보존**: 마이그레이션 시 기존 데이터 유지 필요
