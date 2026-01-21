#!/usr/bin/env python3
"""
HuggingFace 모델 다운로드 유틸리티

Usage:
    # 단일 모델 다운로드
    python -m src.utils.download_model meta-llama/Llama-3.1-8B-Instruct

    # 여러 모델 다운로드 (YAML 설정 파일)
    python -m src.utils.download_model --config models/model_list.yaml

    # 특정 디렉토리에 다운로드
    python -m src.utils.download_model meta-llama/Llama-3.1-8B-Instruct --local-dir ./my_models
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

load_dotenv()

# 기본 설정
DEFAULT_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models/downloaded")


def get_model_local_path(model_id: str, local_dir: str = DEFAULT_CACHE_DIR) -> Path:
    """모델 ID에서 로컬 경로 생성"""
    # meta-llama/Llama-3.1-8B-Instruct -> models/downloaded/meta-llama--Llama-3.1-8B-Instruct
    safe_name = model_id.replace("/", "--")
    return Path(local_dir) / safe_name


def check_model_exists(model_id: str, local_dir: str = DEFAULT_CACHE_DIR) -> bool:
    """로컬에 모델이 이미 다운로드되어 있는지 확인"""
    model_path = get_model_local_path(model_id, local_dir)
    
    # 필수 파일 체크
    required_files = ["config.json"]
    
    if not model_path.exists():
        return False
    
    for filename in required_files:
        if not (model_path / filename).exists():
            return False
    
    return True


def get_model_info(model_id: str, token: Optional[str] = None) -> dict:
    """HuggingFace Hub에서 모델 정보 조회"""
    api = HfApi()
    try:
        info = api.model_info(model_id, token=token)
        return {
            "id": info.id,
            "sha": info.sha,
            "private": info.private,
            "gated": info.gated,
            "downloads": info.downloads,
            "library_name": info.library_name,
            "pipeline_tag": info.pipeline_tag,
        }
    except RepositoryNotFoundError:
        return {"error": f"모델을 찾을 수 없습니다: {model_id}"}
    except GatedRepoError:
        return {"error": f"접근 권한이 필요한 모델입니다: {model_id}", "gated": True}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
    before_sleep=lambda retry_state: print(f"  재시도 중... ({retry_state.attempt_number}/3)")
)
def download_model(
    model_id: str,
    local_dir: str = DEFAULT_CACHE_DIR,
    revision: str = "main",
    token: Optional[str] = None,
    resume_download: bool = True,
    force: bool = False,
    ignore_patterns: Optional[List[str]] = None
) -> Path:
    """
    HuggingFace Hub에서 모델 다운로드

    Args:
        model_id: HuggingFace 모델 ID (예: meta-llama/Llama-3.1-8B-Instruct)
        local_dir: 다운로드 경로 (기본: models/downloaded)
        revision: 브랜치 또는 커밋 해시 (기본: main)
        token: HuggingFace 토큰 (gated 모델용)
        resume_download: 중단된 다운로드 이어받기
        force: 기존 다운로드 무시하고 재다운로드
        ignore_patterns: 제외할 파일 패턴 목록

    Returns:
        다운로드된 모델 경로
    """
    # 토큰 설정
    if token is None:
        token = os.getenv("HUGGINGFACE_TOKEN")
    
    # 로컬 경로 설정
    model_path = get_model_local_path(model_id, local_dir)
    
    # 이미 존재하는지 확인
    if not force and check_model_exists(model_id, local_dir):
        print(f"✓ 모델이 이미 존재합니다: {model_path}")
        return model_path
    
    # 기본 제외 패턴
    if ignore_patterns is None:
        ignore_patterns = [
            "*.md",
            "*.txt",
            ".gitattributes",
            "original/**",  # 원본 체크포인트 제외
        ]
    
    print(f"\n{'='*60}")
    print(f"모델 다운로드: {model_id}")
    print(f"{'='*60}")
    print(f"  저장 경로: {model_path}")
    print(f"  리비전: {revision}")
    print(f"  토큰: {'설정됨' if token else '미설정'}")
    print()
    
    # 다운로드 실행
    downloaded_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(model_path),
        token=token,
        resume_download=resume_download,
        ignore_patterns=ignore_patterns,
    )
    
    print(f"\n✓ 다운로드 완료: {downloaded_path}")
    
    return Path(downloaded_path)


def download_models_from_config(config_path: str, local_dir: str = DEFAULT_CACHE_DIR) -> List[Path]:
    """
    YAML 설정 파일에서 모델 목록을 읽어 다운로드

    설정 파일 형식 (model_list.yaml):
    ```yaml
    models:
      - id: meta-llama/Llama-3.1-8B-Instruct
        revision: main
      - id: mistralai/Mistral-7B-Instruct-v0.2
    ```
    """
    try:
        import yaml
    except ImportError:
        print("✗ PyYAML이 필요합니다: pip install pyyaml")
        sys.exit(1)
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"✗ 설정 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    models = config.get("models", [])
    if not models:
        print("✗ 설정 파일에 모델이 없습니다")
        sys.exit(1)
    
    print(f"\n총 {len(models)}개 모델 다운로드 예정\n")
    
    downloaded_paths = []
    for i, model_config in enumerate(models, 1):
        model_id = model_config.get("id") or model_config.get("model_id")
        revision = model_config.get("revision", "main")
        
        print(f"[{i}/{len(models)}] {model_id}")
        
        try:
            path = download_model(
                model_id=model_id,
                local_dir=local_dir,
                revision=revision,
            )
            downloaded_paths.append(path)
        except Exception as e:
            print(f"  ✗ 다운로드 실패: {e}")
    
    return downloaded_paths


def list_downloaded_models(local_dir: str = DEFAULT_CACHE_DIR) -> List[dict]:
    """다운로드된 모델 목록 조회"""
    local_path = Path(local_dir)
    
    if not local_path.exists():
        return []
    
    models = []
    for model_dir in local_path.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("."):
            config_file = model_dir / "config.json"
            if config_file.exists():
                # 디렉토리 크기 계산
                size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                size_gb = size / (1024 ** 3)
                
                models.append({
                    "name": model_dir.name.replace("--", "/"),
                    "path": str(model_dir),
                    "size_gb": round(size_gb, 2),
                })
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace 모델 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 모델 다운로드
  python -m src.utils.download_model meta-llama/Llama-3.1-8B-Instruct

  # 설정 파일로 여러 모델 다운로드
  python -m src.utils.download_model --config models/model_list.yaml

  # 다운로드된 모델 목록 확인
  python -m src.utils.download_model --list

  # 모델 정보 확인
  python -m src.utils.download_model --info meta-llama/Llama-3.1-8B-Instruct
        """
    )
    
    parser.add_argument(
        "model_id",
        nargs="?",
        help="HuggingFace 모델 ID (예: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--config",
        help="모델 목록 YAML 설정 파일"
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"다운로드 경로 (기본: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="브랜치 또는 커밋 (기본: main)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 다운로드 무시하고 재다운로드"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="다운로드된 모델 목록 출력"
    )
    parser.add_argument(
        "--info",
        metavar="MODEL_ID",
        help="HuggingFace Hub에서 모델 정보 조회"
    )
    
    args = parser.parse_args()
    
    # 다운로드된 모델 목록
    if args.list:
        models = list_downloaded_models(args.local_dir)
        if not models:
            print(f"다운로드된 모델이 없습니다: {args.local_dir}")
        else:
            print(f"\n다운로드된 모델 ({len(models)}개):\n")
            for m in models:
                print(f"  • {m['name']}")
                print(f"    경로: {m['path']}")
                print(f"    크기: {m['size_gb']} GB")
                print()
        return
    
    # 모델 정보 조회
    if args.info:
        token = os.getenv("HUGGINGFACE_TOKEN")
        info = get_model_info(args.info, token=token)
        
        if "error" in info:
            print(f"✗ {info['error']}")
            if info.get("gated"):
                print("  HUGGINGFACE_TOKEN 환경변수를 설정하고 HuggingFace에서 모델 접근을 승인받으세요.")
        else:
            print(f"\n모델 정보: {args.info}\n")
            for key, value in info.items():
                print(f"  {key}: {value}")
        return
    
    # 설정 파일로 다운로드
    if args.config:
        download_models_from_config(args.config, args.local_dir)
        return
    
    # 단일 모델 다운로드
    if args.model_id:
        try:
            download_model(
                model_id=args.model_id,
                local_dir=args.local_dir,
                revision=args.revision,
                force=args.force,
            )
        except RepositoryNotFoundError:
            print(f"✗ 모델을 찾을 수 없습니다: {args.model_id}")
            sys.exit(1)
        except GatedRepoError:
            print(f"✗ 접근 권한이 필요한 모델입니다: {args.model_id}")
            print("  HUGGINGFACE_TOKEN 환경변수를 설정하고 HuggingFace에서 모델 접근을 승인받으세요.")
            sys.exit(1)
        except Exception as e:
            print(f"✗ 다운로드 실패: {e}")
            sys.exit(1)
        return
    
    # 인자가 없으면 도움말 출력
    parser.print_help()


if __name__ == "__main__":
    main()
