# admin/ - SQLAdmin 관리자 인터페이스

> **상위 문서**: [src/serve/CLAUDE.md](../CLAUDE.md) 참조

## 구조

| 파일 | 설명 |
|------|------|
| `__init__.py` | Admin 앱 생성 |
| `auth.py` | JWT 인증 백엔드 |
| `views.py` | 모델 Admin 뷰 |

## 접근

- URL: `/admin`
- 인증: `ADMIN_USERNAME` / `ADMIN_PASSWORD`

## 환경변수

| 변수 | 설명 |
|------|------|
| `ADMIN_USERNAME` | 관리자 ID |
| `ADMIN_PASSWORD` | 관리자 비밀번호 |
| `JWT_SECRET_KEY` | JWT 서명 키 |
