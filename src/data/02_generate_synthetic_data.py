#!/usr/bin/env python3
"""
Phase 2-2: 합성 데이터 생성

GPT-4 또는 다른 LLM을 사용하여 학습용 합성 데이터를 생성합니다.
특정 도메인(예: MLOps, DevOps)에 맞는 대화 데이터를 생성할 수 있습니다.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def generate_with_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    OpenAI API를 사용한 텍스트 생성

    Args:
        prompt: 생성 프롬프트
        model: 사용할 모델 (gpt-3.5-turbo, gpt-4, etc.)

    Returns:
        생성된 텍스트
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates high-quality training data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    except ImportError:
        print("✗ OpenAI library not installed. Run: pip install openai")
        return None
    except Exception as e:
        print(f"✗ Error calling OpenAI API: {e}")
        return None


def generate_mlops_topics() -> List[str]:
    """MLOps 관련 주제 목록 생성"""
    topics = [
        # MLOps 기본
        "What is MLOps and why is it important?",
        "Explain the MLOps lifecycle",
        "What are the key components of an MLOps platform?",
        "Differences between DevOps and MLOps",

        # 모델 개발
        "How to version control machine learning models?",
        "Explain model training pipeline best practices",
        "What is experiment tracking in ML?",
        "How to handle data versioning in ML projects?",

        # 배포
        "What are different model deployment strategies?",
        "Explain A/B testing for ML models",
        "How to implement canary deployments for models?",
        "What is model serving and inference optimization?",

        # 모니터링
        "How to monitor ML models in production?",
        "What is model drift and how to detect it?",
        "Explain data drift vs concept drift",
        "How to set up alerts for model performance degradation?",

        # 도구
        "Compare MLflow, Weights & Biases, and Neptune",
        "How to use DVC for data versioning?",
        "Explain model registry and its importance",
        "What is feature store and when to use it?",

        # 실무
        "How to build a CI/CD pipeline for ML models?",
        "Explain blue-green deployment for ML systems",
        "What are the challenges in productionizing ML models?",
        "How to handle model retraining in production?",

        # LLM 특화
        "How to fine-tune large language models efficiently?",
        "What is LoRA and QLoRA?",
        "Explain prompt engineering best practices",
        "How to optimize LLM inference speed?",

        # GPU/리소스 관리
        "How to optimize GPU memory usage during training?",
        "Explain mixed precision training",
        "What is gradient accumulation?",
        "How to implement efficient batch processing for LLMs?"
    ]

    return topics


def create_qa_pair(topic: str, use_openai: bool = False, model: str = "gpt-3.5-turbo") -> Dict:
    """
    주제에 대한 Q&A 쌍 생성

    Args:
        topic: 주제 (질문)
        use_openai: OpenAI API 사용 여부
        model: OpenAI 모델명

    Returns:
        {"instruction": str, "input": str, "output": str}
    """
    if use_openai:
        # OpenAI API로 답변 생성
        prompt = f"""Generate a detailed, accurate, and helpful answer to the following question about MLOps, DevOps, or machine learning.

Question: {topic}

Provide a comprehensive answer that includes:
1. Clear explanation of concepts
2. Practical examples or use cases
3. Best practices if applicable
4. Common pitfalls to avoid if relevant

Answer:"""

        answer = generate_with_openai(prompt, model)

        if answer is None:
            return None

        return {
            "instruction": topic,
            "input": "",
            "output": answer
        }
    else:
        # 템플릿 기반 답변 (OpenAI 없이 사용 가능)
        return {
            "instruction": topic,
            "input": "",
            "output": f"This is a placeholder answer for: {topic}\n\nPlease enable OpenAI API to generate real answers, or manually add answers to this dataset."
        }


def generate_synthetic_dataset(
    num_examples: int = 30,
    use_openai: bool = False,
    model: str = "gpt-3.5-turbo",
    output_file: str = "data/synthetic_train.json"
) -> List[Dict]:
    """
    합성 데이터셋 생성

    Args:
        num_examples: 생성할 예제 수
        use_openai: OpenAI API 사용 여부
        model: OpenAI 모델명
        output_file: 출력 파일 경로

    Returns:
        생성된 데이터셋
    """
    print(f"\n{'='*60}")
    print("Generating Synthetic Dataset")
    print(f"{'='*60}\n")
    print(f"Examples to generate: {num_examples}")
    print(f"Using OpenAI: {use_openai}")
    if use_openai:
        print(f"Model: {model}")
    print()

    # 주제 생성
    topics = generate_mlops_topics()

    # num_examples보다 주제가 적으면 반복
    if num_examples > len(topics):
        topics = topics * (num_examples // len(topics) + 1)

    topics = topics[:num_examples]

    # Q&A 쌍 생성
    dataset = []
    failed = 0

    for i, topic in enumerate(tqdm(topics, desc="Generating examples"), 1):
        qa_pair = create_qa_pair(topic, use_openai, model)

        if qa_pair is not None:
            dataset.append(qa_pair)

            # OpenAI API rate limit 고려
            if use_openai and i % 10 == 0:
                time.sleep(1)
        else:
            failed += 1

    print(f"\n✓ Generated {len(dataset)} examples")
    if failed > 0:
        print(f"✗ Failed to generate {failed} examples")

    # 데이터 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✓ Dataset saved to: {output_path}")

    # 통계 출력
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total examples: {len(dataset)}")

    if dataset:
        avg_instruction_len = sum(len(ex["instruction"]) for ex in dataset) / len(dataset)
        avg_output_len = sum(len(ex["output"]) for ex in dataset) / len(dataset)

        print(f"Avg instruction length: {avg_instruction_len:.0f} characters")
        print(f"Avg output length: {avg_output_len:.0f} characters")

    print(f"{'='*60}\n")

    return dataset


def preview_dataset(dataset: List[Dict], num_examples: int = 3):
    """데이터셋 미리보기"""
    print(f"\n{'='*60}")
    print(f"Dataset Preview (first {num_examples} examples)")
    print(f"{'='*60}\n")

    for i, example in enumerate(dataset[:num_examples], 1):
        print(f"Example {i}:")
        print("-" * 60)
        print(f"Instruction: {example['instruction']}")
        if example.get('input'):
            print(f"Input: {example['input']}")
        output = example['output']
        if len(output) > 300:
            output = output[:300] + "..."
        print(f"Output: {output}")
        print()


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("  Phase 2-2: Generate Synthetic Data")
    print("="*60 + "\n")

    # OpenAI API 키 확인
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

    if has_openai_key:
        print("✓ OpenAI API key found")
    else:
        print("⚠ OpenAI API key not found")
        print("  Add OPENAI_API_KEY to .env file to use GPT-4/3.5")
        print("  Or continue with template-based generation")

    print()

    # 설정
    num_examples = input("Number of examples to generate (default: 30): ").strip()
    num_examples = int(num_examples) if num_examples else 30

    use_openai = False
    model = "gpt-3.5-turbo"

    if has_openai_key:
        use_choice = input("Use OpenAI API? (y/n, default: n): ").strip().lower()
        use_openai = (use_choice == "y")

        if use_openai:
            print("\nAvailable models:")
            print("  1. gpt-3.5-turbo (faster, cheaper)")
            print("  2. gpt-4 (better quality, more expensive)")
            model_choice = input("Select model (1-2, default: 1): ").strip()
            model = "gpt-4" if model_choice == "2" else "gpt-3.5-turbo"

    output_file = input("\nOutput file (default: data/synthetic_train.json): ").strip()
    output_file = output_file if output_file else "data/synthetic_train.json"

    # 데이터셋 생성
    try:
        dataset = generate_synthetic_dataset(
            num_examples=num_examples,
            use_openai=use_openai,
            model=model,
            output_file=output_file
        )

        # 미리보기
        preview_dataset(dataset)

        print("✓ Synthetic data generation completed!")
        print("\nNext steps:")
        print("  1. Review generated data: cat data/synthetic_train.json")
        print("  2. Start fine-tuning: python src/train/01_lora_finetune.py")

    except Exception as e:
        print(f"\n✗ Error generating dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
