#!/usr/bin/env python3
"""
Phase 1-1: 베이스 모델 로드 및 추론 테스트

사전학습된 LLM을 로드하고 기본 추론을 테스트합니다.
GPU 메모리 사용량과 추론 속도를 측정합니다.
"""

import os
import time
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

# .env 파일 로드
load_dotenv()


def get_device():
    """사용 가능한 디바이스 확인"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("✓ Using Apple MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("⚠ Using CPU (slow, not recommended for production)")

    return device


def load_model_basic(model_name, device):
    """
    기본 방식으로 모델 로드 (Full precision)
    GPU 메모리가 충분한 경우 사용
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Mode: Full Precision (FP32/FP16)")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Tokenizer 로드
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )

    # 특수 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )

    if device != "cuda":
        model = model.to(device)

    load_time = time.time() - start_time
    print(f"\n✓ Model loaded in {load_time:.2f} seconds")

    # 메모리 사용량 출력
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU Memory Allocated: {allocated:.2f} GB")
        print(f"  GPU Memory Reserved: {reserved:.2f} GB")

    return model, tokenizer


def load_model_quantized(model_name, device):
    """
    양자화 방식으로 모델 로드 (4-bit)
    GPU 메모리가 부족한 경우 사용
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Mode: 4-bit Quantization")
    print(f"{'='*60}\n")

    if device != "cuda":
        raise ValueError("Quantization requires CUDA GPU")

    start_time = time.time()

    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Tokenizer 로드
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN")
    )

    load_time = time.time() - start_time
    print(f"\n✓ Model loaded in {load_time:.2f} seconds")

    # 메모리 사용량 출력
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  GPU Memory Allocated: {allocated:.2f} GB")
    print(f"  GPU Memory Reserved: {reserved:.2f} GB")

    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
    """텍스트 생성 및 성능 측정"""
    print(f"\n{'='*60}")
    print("Generating response...")
    print(f"{'='*60}\n")

    print(f"Prompt: {prompt}\n")

    # 입력 준비
    inputs = tokenizer(prompt, return_tensors="pt")

    if device == "cuda":
        inputs = inputs.to("cuda")
    elif device == "mps":
        inputs = inputs.to("mps")

    # 생성 시작
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generation_time = time.time() - start_time

    # 결과 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()

    # 성능 메트릭 계산
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / generation_time

    print(f"Response: {response}\n")
    print(f"{'='*60}")
    print(f"Performance Metrics:")
    print(f"  Generation Time: {generation_time:.2f} seconds")
    print(f"  Tokens Generated: {num_tokens}")
    print(f"  Tokens/Second: {tokens_per_sec:.2f}")
    print(f"{'='*60}\n")

    return response, {
        "generation_time": generation_time,
        "num_tokens": num_tokens,
        "tokens_per_sec": tokens_per_sec
    }


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("  Phase 1-1: Base Model Test")
    print("="*60 + "\n")

    # 디바이스 확인
    device = get_device()

    # 모델 이름 설정 (.env에서 읽거나 기본값 사용)
    model_name = os.getenv("BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

    # 사용자 선택
    print("\nSelect loading mode:")
    print("  1) Full Precision (requires ~14GB VRAM)")
    print("  2) 4-bit Quantization (requires ~4GB VRAM)")
    print("  3) Skip (if already loaded)")

    if device != "cuda":
        print("\n⚠ No CUDA GPU detected. Only full precision on CPU/MPS available.")
        mode = "1"
    else:
        mode = input("\nEnter choice (1-3): ").strip()

    # 모델 로드
    try:
        if mode == "1":
            model, tokenizer = load_model_basic(model_name, device)
        elif mode == "2":
            if device != "cuda":
                print("✗ Error: Quantization requires CUDA GPU")
                return
            model, tokenizer = load_model_quantized(model_name, device)
        elif mode == "3":
            print("Skipping model load")
            return
        else:
            print("Invalid choice")
            return

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check HuggingFace token in .env")
        print("  2. Verify model name")
        print("  3. Check GPU memory availability")
        print("  4. Try 4-bit quantization mode")
        return

    # 테스트 프롬프트
    test_prompts = [
        "What is machine learning?",
        "Explain MLOps in simple terms.",
        "Write a Python function to calculate fibonacci numbers."
    ]

    # 각 프롬프트 테스트
    results = []
    for prompt in test_prompts:
        response, metrics = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100,
            device=device
        )
        results.append(metrics)

    # 평균 성능 출력
    avg_time = sum(r["generation_time"] for r in results) / len(results)
    avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in results) / len(results)

    print(f"{'='*60}")
    print("Average Performance:")
    print(f"  Avg Generation Time: {avg_time:.2f} seconds")
    print(f"  Avg Tokens/Second: {avg_tokens_per_sec:.2f}")
    print(f"{'='*60}\n")

    print("✓ Test completed!")
    print("\nNext steps:")
    print("  1. Run Gradio demo: python src/02_gradio_demo.py")
    print("  2. Run benchmark: python src/03_benchmark.py")


if __name__ == "__main__":
    main()
