# src/utils/ - 유틸리티

> **상위 문서**: [루트 CLAUDE.md](../../CLAUDE.md) 참조

구조화된 로깅, GPU 모니터링, 모델 다운로드

## 파일

| 파일 | 설명 |
|------|------|
| `logging_utils.py` | JSON 로깅 (structlog) |
| `gpu_monitor.py` | GPU 메트릭 |
| `download_model.py` | HuggingFace 모델 다운로드 |
| `__init__.py` | 패키지 |

## 모델 다운로드

### 사용법

```bash
# 단일 모델 다운로드
python -m src.utils.download_model meta-llama/Llama-3.1-8B-Instruct

# 다운로드된 모델 목록 확인
python -m src.utils.download_model --list

# 모델 정보 조회
python -m src.utils.download_model --info meta-llama/Llama-3.1-8B-Instruct

# 설정 파일로 여러 모델 다운로드
python -m src.utils.download_model --config models/model_list.yaml
```

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_CACHE_DIR` | models/downloaded | 모델 캐시 경로 |
| `OFFLINE_MODE` | false | 오프라인 모드 |
| `HUGGINGFACE_TOKEN` | - | Gated 모델 접근 토큰 |

### 주요 함수

```python
from src.utils.download_model import download_model, check_model_exists

# 모델 다운로드
path = download_model("meta-llama/Llama-3.1-8B-Instruct")

# 이미 존재하는지 확인
exists = check_model_exists("meta-llama/Llama-3.1-8B-Instruct")
```

## Import 경로

```python
# 프로젝트 루트에서 실행 시
from src.utils.logging_utils import TrainingLogger, LogType

# src/ 내부에서 실행 시 (상대 경로)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.logging_utils import TrainingLogger
```

## 로그 타입

```python
class LogType:
    TRAINING = "training"
    INFERENCE = "inference"
    SYSTEM = "system"
    API = "api"
```

## 로거 클래스 및 주요 메서드

### TrainingLogger
```python
logger = TrainingLogger("experiment-name")
logger.log_epoch_start(epoch=1, total_epochs=3)
logger.log_step(epoch=1, step=100, loss=0.5, learning_rate=2e-4)
logger.log_epoch_end(epoch=1, avg_loss=0.45)
logger.log_validation(epoch=1, val_loss=0.48)
logger.log_error("OOM 발생")
```

### InferenceLogger
```python
logger = InferenceLogger("vllm-server")
logger.log_request(request_id="abc", prompt="질문")
logger.log_response(request_id="abc", latency_ms=150, tokens_generated=50)
logger.log_error(request_id="abc", error="타임아웃")
```

### SystemLogger
```python
logger = SystemLogger("system-monitor")
logger.log_gpu_metrics(gpu_id=0, gpu_memory_used=8000, gpu_memory_total=31000, gpu_utilization=80.5)
logger.log_system_metrics(cpu_percent=45.0, memory_percent=60.0, disk_percent=30.0)
```

### APILogger
```python
logger = APILogger("fastapi")
logger.log_request(request_id="xyz", method="POST", path="/v1/chat")
logger.log_response(request_id="xyz", status_code=200, duration_ms=250)
```

## 로그 저장 위치

```
logs/
├── training/    # 학습 (epoch, loss)
├── inference/   # 추론 (latency, tokens/sec)
├── system/      # GPU, CPU, 메모리
└── api/         # HTTP 요청/응답
```

## Loki/Grafana 연동

JSON 포맷 → Promtail → Loki → Grafana
