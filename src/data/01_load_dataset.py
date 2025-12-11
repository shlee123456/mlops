#!/usr/bin/env python3
"""
Phase 2-1: 데이터셋 로드

공개 데이터셋을 다운로드하고 탐색합니다.
Fine-tuning을 위한 데이터 형식을 확인합니다.
"""

import os
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def explore_dataset(dataset, num_examples=5):
    """데이터셋 구조 탐색"""
    print(f"\n{'='*60}")
    print("Dataset Structure")
    print(f"{'='*60}\n")

    # 데이터셋 크기
    if hasattr(dataset, 'num_rows'):
        print(f"Number of examples: {dataset.num_rows:,}")
    else:
        print(f"Number of splits: {len(dataset)}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data):,} examples")

    # 첫 번째 예제 확인
    print(f"\n{'='*60}")
    print("First Example")
    print(f"{'='*60}\n")

    if hasattr(dataset, 'features'):
        example = dataset[0]
        features = dataset.features
    else:
        # Split이 있는 경우 train 사용
        split_name = list(dataset.keys())[0]
        example = dataset[split_name][0]
        features = dataset[split_name].features

    print(f"Features: {list(features.keys())}")
    print(f"\nExample:")
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...")
        else:
            print(f"  {key}: {value}")

    # 여러 예제 출력
    print(f"\n{'='*60}")
    print(f"Sample Examples (first {num_examples})")
    print(f"{'='*60}\n")

    dataset_to_use = dataset if hasattr(dataset, 'features') else dataset[split_name]

    for i in range(min(num_examples, len(dataset_to_use))):
        example = dataset_to_use[i]
        print(f"\nExample {i+1}:")
        print("-" * 60)
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 150:
                print(f"{key}: {value[:150]}...")
            else:
                print(f"{key}: {value}")


def load_and_explore_datasets():
    """여러 공개 데이터셋 로드 및 탐색"""

    # 추천 데이터셋 목록
    datasets_info = {
        "1": {
            "name": "HuggingFaceH4/no_robots",
            "description": "High-quality instruction following dataset",
            "size": "~10k examples"
        },
        "2": {
            "name": "tatsu-lab/alpaca",
            "description": "Stanford Alpaca instruction dataset",
            "size": "~52k examples"
        },
        "3": {
            "name": "timdettmers/openassistant-guanaco",
            "description": "OpenAssistant conversations",
            "size": "~10k examples"
        },
        "4": {
            "name": "databricks/databricks-dolly-15k",
            "description": "Databricks Dolly instruction dataset",
            "size": "~15k examples"
        }
    }

    print("\n" + "="*60)
    print("  Phase 2-1: Load Dataset")
    print("="*60 + "\n")

    print("Available datasets:")
    for key, info in datasets_info.items():
        print(f"\n{key}. {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")

    print("\n" + "="*60)
    choice = input("\nSelect dataset (1-4) or enter custom name: ").strip()

    # 데이터셋 이름 결정
    if choice in datasets_info:
        dataset_name = datasets_info[choice]["name"]
    else:
        dataset_name = choice

    print(f"\nLoading dataset: {dataset_name}")
    print("This may take a few minutes...\n")

    try:
        # 데이터셋 로드
        dataset = load_dataset(dataset_name)

        print("✓ Dataset loaded successfully!\n")

        # 데이터셋 탐색
        explore_dataset(dataset, num_examples=3)

        # 데이터 저장 옵션
        print(f"\n{'='*60}")
        save_choice = input("\nSave dataset locally? (y/n): ").strip().lower()

        if save_choice == "y":
            # 저장 경로
            data_dir = Path("data/raw")
            data_dir.mkdir(parents=True, exist_ok=True)

            # 데이터셋 이름에서 경로 분리
            dataset_short_name = dataset_name.split("/")[-1]
            save_path = data_dir / dataset_short_name

            print(f"\nSaving to: {save_path}")

            # 데이터셋 저장
            if isinstance(dataset, dict):
                for split_name, split_data in dataset.items():
                    split_path = save_path / split_name
                    split_data.save_to_disk(str(split_path))
            else:
                dataset.save_to_disk(str(save_path))

            print("✓ Dataset saved!")

        return dataset

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify dataset name")
        print("  3. Check if HuggingFace token is required")
        print("  4. Try a different dataset")
        return None


def format_examples_for_training(dataset, output_file="data/processed/formatted_data.jsonl"):
    """
    데이터를 학습용 형식으로 변환

    일반적인 instruction-following 형식:
    {
        "instruction": "질문 또는 지시사항",
        "input": "추가 입력 (선택)",
        "output": "응답"
    }
    """
    import json
    from tqdm import tqdm

    print(f"\n{'='*60}")
    print("Formatting data for training")
    print(f"{'='*60}\n")

    # Split 선택
    if isinstance(dataset, dict):
        print("Available splits:", list(dataset.keys()))
        split_name = input("Select split to format (e.g., 'train'): ").strip()
        if split_name not in dataset:
            print(f"✗ Split '{split_name}' not found")
            return
        data = dataset[split_name]
    else:
        data = dataset

    # 필드 매핑 확인
    features = list(data.features.keys())
    print(f"Available fields: {features}\n")

    print("Map fields to instruction format:")
    instruction_field = input("Instruction field (e.g., 'prompt', 'question'): ").strip()
    output_field = input("Output field (e.g., 'response', 'answer'): ").strip()
    input_field = input("Input field (optional, press Enter to skip): ").strip() or None

    # 출력 경로 설정
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 변환 및 저장
    print(f"\nFormatting {len(data)} examples...")

    formatted_count = 0
    with open(output_path, "w") as f:
        for example in tqdm(data):
            try:
                formatted = {
                    "instruction": example.get(instruction_field, ""),
                    "input": example.get(input_field, "") if input_field else "",
                    "output": example.get(output_field, "")
                }

                # 비어있는 예제 제외
                if formatted["instruction"] and formatted["output"]:
                    f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                    formatted_count += 1

            except Exception as e:
                print(f"Warning: Failed to format example: {e}")
                continue

    print(f"\n✓ Formatted {formatted_count} examples")
    print(f"✓ Saved to: {output_path}")

    return output_path


def main():
    """메인 실행 함수"""
    # 데이터셋 로드 및 탐색
    dataset = load_and_explore_datasets()

    if dataset is None:
        return

    # 데이터 포맷팅 옵션
    print(f"\n{'='*60}")
    format_choice = input("\nFormat data for training? (y/n): ").strip().lower()

    if format_choice == "y":
        format_examples_for_training(dataset)

    print(f"\n{'='*60}")
    print("Next steps:")
    print("  1. Generate synthetic data: python src/data/02_generate_synthetic_data.py")
    print("  2. Start training: python src/train/01_lora_finetune.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
