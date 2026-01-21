# admin/ - SQLAdmin 관리자 인터페이스

> **상위 문서**: [src/serve/CLAUDE.md](../CLAUDE.md) 참조

## 구조

```
admin/
├── __init__.py      # Admin 앱 생성 및 뷰 등록
├── auth.py          # JWT 인증 백엔드
├── views.py         # ModelView 및 BaseView 정의
└── templates/       # 커스텀 HTML 템플릿
    ├── system_status.html
    └── message_statistics.html
```

| 파일 | 설명 |
|------|------|
| `__init__.py` | Admin 앱 생성, templates_dir 설정 |
| `auth.py` | JWT 인증 백엔드 |
| `views.py` | ModelView (3개) + BaseView (3개) |
| `templates/` | 커스텀 대시보드 템플릿 |

## 기능

### ModelView (데이터 CRUD)

| 클래스 | 모델 | 기능 |
|--------|------|------|
| `LLMConfigAdmin` | LLMConfig | LLM 설정 관리 |
| `ConversationAdmin` | Conversation | 대화 관리 + 커스텀 검색 |
| `ChatMessageAdmin` | ChatMessage | 메시지 조회 (읽기 전용) + 커스텀 검색 |

### 커스텀 검색 (search_query)

- **ConversationAdmin**: ID 숫자 검색, title/session_id ilike 검색
- **ChatMessageAdmin**: ID/conversation_id 숫자 검색, content/role ilike 검색

### BaseView (커스텀 대시보드)

| 클래스 | 경로 | 설명 |
|--------|------|------|
| `SystemStatusView` | `/admin/system-status` | 시스템 상태 (메모리, CPU, 디스크, DB 풀) |
| `MessageStatisticsView` | `/admin/message-statistics` | 메시지 통계 (날짜 필터, 역할별/일별 통계) |
| `VLLMStatusView` | `/admin/vllm-status` | vLLM /metrics 리다이렉트 |

## 접근

- URL: `/admin`
- 로그인: `/admin/login`
- 인증: `ADMIN_USERNAME` / `ADMIN_PASSWORD`

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ADMIN_USERNAME` | admin | 관리자 ID |
| `ADMIN_PASSWORD` | changeme | 관리자 비밀번호 |
| `JWT_SECRET_KEY` | change-this-... | JWT 서명 키 |
| `JWT_ALGORITHM` | HS256 | JWT 알고리즘 |
| `ADMIN_TOKEN_EXPIRE_MINUTES` | 60 | 토큰 만료 시간 |

## 의존성

- `sqladmin>=0.16.0` - Admin UI 프레임워크
- `python-jose[cryptography]>=3.3.0` - JWT 처리
- `passlib[bcrypt]>=1.7.4` - 비밀번호 해싱
- `psutil>=5.9.0` - 시스템 모니터링

## 테스트

```bash
python -m pytest tests/serve/test_admin.py -v
```

### 테스트 항목 (16개)

- JWT 토큰 생성/검증 (4개)
- Admin 페이지 접근 (5개)
- 커스텀 검색 (3개)
- BaseView 대시보드 (4개)
