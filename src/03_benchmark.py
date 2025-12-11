#!/usr/bin/env python3
"""
Phase 1-3: 성능 벤치마크

다양한 시나리오에서 모델 성능을 측정하고 비교합니다.
- Latency (응답 시간)
- Throughput (처리량)
- Memory Usage (메모리 사용량)
"""

import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# .env 파일 로드
load_dotenv()


class ModelBenchmark:
    """모델 벤치마크 클래스"""

    def __init__(self, model_name, device="cuda", use_quantization=False):
        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """모델 로드"""
        print(f"\nLoading model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Quantization: {self.use_quantization}")

        start_time = time.time()

        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        if self.use_quantization and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds\n")

        return load_time

    def get_memory_usage(self):
        """메모리 사용량 측정"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved
            }
        return {"allocated_gb": 0, "reserved_gb": 0}

    def generate_single(self, prompt, max_new_tokens=100):
        """단일 생성 및 시간 측정"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.device == "cuda":
            inputs = inputs.to("cuda")
        elif self.device == "mps":
            inputs = inputs.to("mps")

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generation_time = time.time() - start_time
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]

        return {
            "generation_time": generation_time,
            "num_tokens": num_tokens,
            "tokens_per_sec": num_tokens / generation_time
        }

    def benchmark_latency(self, prompts, max_new_tokens=100, num_runs=5):
        """레이턴시 벤치마크"""
        print(f"{'='*60}")
        print("Benchmark 1: Latency Test")
        print(f"{'='*60}")
        print(f"Num prompts: {len(prompts)}")
        print(f"Runs per prompt: {num_runs}")
        print(f"Max tokens: {max_new_tokens}\n")

        results = []

        for i, prompt in enumerate(prompts, 1):
            print(f"Testing prompt {i}/{len(prompts)}: {prompt[:50]}...")

            run_results = []
            for run in range(num_runs):
                result = self.generate_single(prompt, max_new_tokens)
                run_results.append(result)

            # 통계 계산
            times = [r["generation_time"] for r in run_results]
            tokens_per_sec = [r["tokens_per_sec"] for r in run_results]

            stats = {
                "prompt": prompt,
                "avg_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "avg_tokens_per_sec": np.mean(tokens_per_sec),
                "num_tokens": run_results[0]["num_tokens"]
            }

            results.append(stats)

            print(f"  Avg time: {stats['avg_time']:.2f}s ± {stats['std_time']:.2f}s")
            print(f"  Avg tokens/sec: {stats['avg_tokens_per_sec']:.2f}\n")

        return results

    def benchmark_throughput(self, prompt, max_new_tokens_list=[50, 100, 200, 500]):
        """처리량 벤치마크 (다양한 길이)"""
        print(f"{'='*60}")
        print("Benchmark 2: Throughput Test")
        print(f"{'='*60}")
        print(f"Token lengths: {max_new_tokens_list}\n")

        results = []

        for max_tokens in max_new_tokens_list:
            print(f"Testing with max_tokens={max_tokens}...")

            result = self.generate_single(prompt, max_tokens)

            stats = {
                "max_tokens": max_tokens,
                "actual_tokens": result["num_tokens"],
                "generation_time": result["generation_time"],
                "tokens_per_sec": result["tokens_per_sec"]
            }

            results.append(stats)

            print(f"  Time: {stats['generation_time']:.2f}s")
            print(f"  Tokens/sec: {stats['tokens_per_sec']:.2f}\n")

        return results

    def benchmark_memory(self):
        """메모리 사용량 벤치마크"""
        print(f"{'='*60}")
        print("Benchmark 3: Memory Usage")
        print(f"{'='*60}\n")

        if self.device != "cuda":
            print("Memory benchmark only available for CUDA devices")
            return {}

        # 초기 메모리
        torch.cuda.reset_peak_memory_stats()
        initial_memory = self.get_memory_usage()

        print(f"Initial Memory:")
        print(f"  Allocated: {initial_memory['allocated_gb']:.2f} GB")
        print(f"  Reserved: {initial_memory['reserved_gb']:.2f} GB\n")

        # 추론 실행
        prompt = "Explain machine learning in detail."
        self.generate_single(prompt, max_new_tokens=200)

        # 피크 메모리
        peak_allocated = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9

        print(f"Peak Memory:")
        print(f"  Allocated: {peak_allocated:.2f} GB")
        print(f"  Reserved: {peak_reserved:.2f} GB\n")

        return {
            "initial_allocated_gb": initial_memory['allocated_gb'],
            "initial_reserved_gb": initial_memory['reserved_gb'],
            "peak_allocated_gb": peak_allocated,
            "peak_reserved_gb": peak_reserved
        }


def save_results(results, filename="benchmark_results.json"):
    """결과를 JSON 파일로 저장"""
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{timestamp}_{filename}"

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")
    return filepath


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("  Phase 1-3: Performance Benchmark")
    print("="*60 + "\n")

    # 디바이스 확인
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("✓ Using Apple MPS")
    else:
        device = "cpu"
        print("⚠ Using CPU")

    # 모델 설정
    model_name = os.getenv("BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

    # 양자화 옵션
    use_quantization = False
    if device == "cuda":
        choice = input("\nUse 4-bit quantization? (y/n): ").strip().lower()
        use_quantization = (choice == "y")

    # 벤치마크 인스턴스 생성
    benchmark = ModelBenchmark(model_name, device, use_quantization)

    # 모델 로드
    try:
        load_time = benchmark.load_model()
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return

    # 테스트 프롬프트
    test_prompts = [
        "What is artificial intelligence?",
        "Explain the concept of neural networks.",
        "Write a Python function to calculate prime numbers.",
        "What are the key principles of DevOps?"
    ]

    # 벤치마크 실행
    all_results = {
        "model_name": model_name,
        "device": device,
        "use_quantization": use_quantization,
        "timestamp": datetime.now().isoformat(),
        "load_time": load_time
    }

    # 1. 레이턴시 벤치마크
    try:
        latency_results = benchmark.benchmark_latency(test_prompts, max_new_tokens=100, num_runs=3)
        all_results["latency"] = latency_results
    except Exception as e:
        print(f"✗ Latency benchmark failed: {e}")

    # 2. 처리량 벤치마크
    try:
        throughput_results = benchmark.benchmark_throughput(
            "Explain machine learning in detail.",
            max_new_tokens_list=[50, 100, 200]
        )
        all_results["throughput"] = throughput_results
    except Exception as e:
        print(f"✗ Throughput benchmark failed: {e}")

    # 3. 메모리 벤치마크
    try:
        memory_results = benchmark.benchmark_memory()
        all_results["memory"] = memory_results
    except Exception as e:
        print(f"✗ Memory benchmark failed: {e}")

    # 결과 저장
    save_results(all_results)

    # 요약 출력
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Quantization: {use_quantization}")
    print(f"Load Time: {load_time:.2f}s")

    if "latency" in all_results:
        avg_latency = np.mean([r["avg_time"] for r in latency_results])
        avg_throughput = np.mean([r["avg_tokens_per_sec"] for r in latency_results])
        print(f"Avg Latency: {avg_latency:.2f}s")
        print(f"Avg Throughput: {avg_throughput:.2f} tokens/sec")

    print(f"{'='*60}\n")

    print("✓ Benchmark completed!")


if __name__ == "__main__":
    main()
