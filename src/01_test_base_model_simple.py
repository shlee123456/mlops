#!/usr/bin/env python3
"""
LLaMA 3-70B 간단 로딩 스크립트 (Multi-GPU 4-bit)
"""

import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()

def main():
    print("="*60)
    print("  LLaMA 3-70B Simple Loading (4-bit Multi-GPU)")
    print("="*60)

    # GPU 확인
    num_gpus = torch.cuda.device_count()
    print(f"\nGPUs detected: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

    # 4-bit 양자화 설정 (간단 버전)
    print("\nSetting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Tokenizer 로드
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드 (자동 multi-GPU 분산)
    print("\nLoading model (this will take 2-5 minutes)...")
    print("Model will be automatically distributed across GPUs...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # 자동 분산
        token=os.getenv("HUGGINGFACE_TOKEN"),
        trust_remote_code=True
    )

    print("\n✓ Model loaded successfully!")

    # 메모리 사용량 확인
    print("\nGPU Memory Usage:")
    for i in range(num_gpus):
        mem = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {mem:.2f} GB")

    # 간단한 테스트
    print("\n" + "="*60)
    print("Testing inference...")
    print("="*60)

    prompt = "What is machine learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"\nPrompt: {prompt}")
    print("Generating response...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse: {response}\n")

    print("✓ Test completed successfully!")

if __name__ == "__main__":
    main()
