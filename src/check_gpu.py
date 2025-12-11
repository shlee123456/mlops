#!/usr/bin/env python3
"""
GPU 환경 확인 스크립트
CUDA, PyTorch, 시스템 정보를 확인합니다.
"""

import platform
import sys
import subprocess
from pathlib import Path


def print_section(title):
    """섹션 헤더 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_python():
    """Python 버전 확인"""
    print_section("Python Environment")
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")


def check_system():
    """시스템 정보 확인"""
    print_section("System Information")
    print(f"OS: {platform.system()}")
    print(f"OS Version: {platform.version()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")


def check_cuda():
    """CUDA 설치 확인"""
    print_section("CUDA Information")

    try:
        # nvidia-smi 실행
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("✓ NVIDIA GPU detected!")
            print("\n" + result.stdout)
            return True
        else:
            print("✗ nvidia-smi failed to run")
            print(f"Error: {result.stderr}")
            return False

    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("NVIDIA drivers may not be installed")
        return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_pytorch():
    """PyTorch 설치 및 CUDA 지원 확인"""
    print_section("PyTorch Information")

    try:
        import torch

        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")

            # 각 GPU 정보 출력
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")

                # GPU 메모리 정보
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                print(f"  Total Memory: {total_memory:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")

                # 현재 메모리 사용량
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  Allocated Memory: {allocated:.2f} GB")
                print(f"  Reserved Memory: {reserved:.2f} GB")

            return True
        else:
            print("✗ CUDA not available in PyTorch")

            # MPS (Apple Silicon) 확인
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("✓ Apple MPS (Metal Performance Shaders) available")
                print("  Note: MPS can be used on Apple Silicon Macs")

            return False

    except ImportError:
        print("✗ PyTorch not installed")
        print("Run: pip install torch")
        return False
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False


def check_transformers():
    """Transformers 라이브러리 확인"""
    print_section("Transformers Library")

    try:
        import transformers
        print(f"✓ Transformers Version: {transformers.__version__}")
        return True
    except ImportError:
        print("✗ Transformers not installed")
        print("Run: pip install transformers")
        return False


def check_accelerate():
    """Accelerate 라이브러리 확인"""
    print_section("Accelerate Library")

    try:
        import accelerate
        print(f"✓ Accelerate Version: {accelerate.__version__}")
        return True
    except ImportError:
        print("✗ Accelerate not installed")
        print("Run: pip install accelerate")
        return False


def check_disk_space():
    """디스크 여유 공간 확인"""
    print_section("Disk Space")

    try:
        import shutil

        # 현재 디렉토리의 디스크 사용량
        total, used, free = shutil.disk_usage(Path.cwd())

        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)

        print(f"Total: {total_gb:.2f} GB")
        print(f"Used: {used_gb:.2f} GB")
        print(f"Free: {free_gb:.2f} GB")

        # 최소 50GB 필요
        if free_gb < 50:
            print("\n⚠ Warning: Less than 50GB free space")
            print("  Recommended: 50GB+ for model downloads")
            return False
        else:
            print("\n✓ Sufficient disk space available")
            return True

    except Exception as e:
        print(f"✗ Error checking disk space: {e}")
        return False


def print_recommendations(has_cuda, has_torch, has_transformers, has_accelerate):
    """설치 권장사항 출력"""
    print_section("Recommendations")

    if not has_cuda:
        print("\n⚠ CUDA not available")
        print("Options:")
        print("1. Install NVIDIA drivers (Linux)")
        print("2. Use CPU (slower, not recommended for training)")
        print("3. Use cloud GPU (Google Colab, AWS, etc.)")

        if platform.system() == "Darwin":  # macOS
            print("4. Use Apple MPS on Apple Silicon (limited support)")

    if not has_torch:
        print("\n⚠ Install PyTorch:")
        if has_cuda:
            print("With CUDA 11.8:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("CPU only:")
            print("  pip install torch")

    if not has_transformers:
        print("\n⚠ Install Transformers:")
        print("  pip install transformers")

    if not has_accelerate:
        print("\n⚠ Install Accelerate:")
        print("  pip install accelerate")

    if has_cuda and has_torch and has_transformers and has_accelerate:
        print("\n✓ All core dependencies installed!")
        print("You're ready to start the project!")


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("  MLOps Chatbot Project - GPU Environment Check")
    print("=" * 60)

    # 각 체크 실행
    check_python()
    check_system()
    has_cuda = check_cuda()
    has_torch = check_pytorch()
    has_transformers = check_transformers()
    has_accelerate = check_accelerate()
    has_space = check_disk_space()

    # 권장사항 출력
    print_recommendations(has_cuda, has_torch, has_transformers, has_accelerate)

    # 최종 상태 출력
    print_section("Summary")
    status = []
    status.append(("CUDA", "✓" if has_cuda else "✗"))
    status.append(("PyTorch", "✓" if has_torch else "✗"))
    status.append(("Transformers", "✓" if has_transformers else "✗"))
    status.append(("Accelerate", "✓" if has_accelerate else "✗"))
    status.append(("Disk Space", "✓" if has_space else "⚠"))

    for name, symbol in status:
        print(f"{symbol} {name}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
