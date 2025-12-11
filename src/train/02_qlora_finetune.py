#!/usr/bin/env python3
"""
Phase 2-4: QLoRA Fine-tuning

4-bit 양자화를 사용한 메모리 효율적인 학습
LoRA보다 ~75% 적은 메모리로 학습 가능
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

load_dotenv()


def load_training_data(data_path: str):
    """학습 데이터 로드"""
    print(f"\n{'='*60}")
    print("Loading Training Data")
    print(f"{'='*60}\n")

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r") as f:
        if data_path.suffix == ".json":
            data = json.load(f)
        elif data_path.suffix == ".jsonl":
            data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    print(f"✓ Loaded {len(data)} examples")

    dataset = Dataset.from_list(data)
    print(f"✓ Dataset created with {len(dataset)} examples")

    return dataset


def format_instruction(example):
    """Instruction 형식으로 포맷팅"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": prompt}


def prepare_dataset(dataset, tokenizer, max_length=512):
    """데이터셋 전처리"""
    print(f"\n{'='*60}")
    print("Preparing Dataset")
    print(f"{'='*60}\n")

    print("Formatting examples...")
    dataset = dataset.map(format_instruction)

    print("Tokenizing examples...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    print(f"✓ Dataset prepared with {len(tokenized_dataset)} examples")

    return tokenized_dataset


def setup_qlora_model(
    model_name,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05
):
    """QLoRA 설정 및 모델 준비 (4-bit 양자화)"""
    print(f"\n{'='*60}")
    print("Setting up QLoRA Model (4-bit)")
    print(f"{'='*60}\n")

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA GPU")

    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

    # Tokenizer 로드
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # 4-bit 양자화 설정
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 모델 로드
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
        trust_remote_code=True
    )

    # 메모리 사용량 출력
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\nInitial GPU Memory:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")

    # k-bit training 준비
    print("\nPreparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    print(f"Configuring LoRA...")
    print(f"  r (rank): {lora_r}")
    print(f"  alpha: {lora_alpha}")
    print(f"  dropout: {lora_dropout}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # LoRA 모델 생성
    model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 수
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"\n✓ QLoRA model ready")
    print(f"  Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"  Total params: {total_params:,}")
    print(f"  Memory efficient: ~75% less than full LoRA")

    return model, tokenizer


def train_model(
    model,
    tokenizer,
    train_dataset,
    output_dir="models/fine-tuned/qlora-mistral-custom",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_length=512,
    use_mlflow=True
):
    """모델 학습"""
    print(f"\n{'='*60}")
    print("Training Model with QLoRA")
    print(f"{'='*60}\n")

    # MLflow 설정
    if use_mlflow and HAS_MLFLOW:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "chatbot-finetuning")
        mlflow.set_experiment(experiment_name)

        run_name = f"qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        mlflow.log_params({
            "model_name": model.config._name_or_path,
            "quantization": "4-bit",
            "method": "QLoRA",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "train_samples": len(train_dataset)
        })

        print(f"✓ MLflow run started: {run_name}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_32bit",  # QLoRA에 최적화된 optimizer
        report_to="mlflow" if (use_mlflow and HAS_MLFLOW) else "none",
        push_to_hub=False,
        disable_tqdm=False
    )

    print(f"Training configuration:")
    print(f"  Output dir: {output_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: paged_adamw_32bit (QLoRA optimized)")
    print()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # 학습 시작
    print("Starting training...\n")
    train_result = trainer.train()

    # 결과 출력
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

    # 최종 메모리 사용량
    allocated = torch.cuda.memory_allocated() / 1e9
    peak_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nGPU Memory Usage:")
    print(f"  Current: {allocated:.2f} GB")
    print(f"  Peak: {peak_allocated:.2f} GB")

    # 모델 저장
    print(f"\nSaving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # MLflow 로깅
    if use_mlflow and HAS_MLFLOW:
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics['train_runtime'],
            "train_samples_per_second": train_result.metrics['train_samples_per_second'],
            "peak_memory_gb": peak_allocated
        })

        mlflow.log_artifact(output_dir)
        mlflow.end_run()
        print("✓ MLflow run completed")

    print(f"\n✓ Model saved successfully!")

    return trainer


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("  Phase 2-4: QLoRA Fine-tuning (4-bit)")
    print("="*60 + "\n")

    # CUDA 확인
    if not torch.cuda.is_available():
        print("✗ Error: QLoRA requires CUDA GPU")
        print("\nOptions:")
        print("  1. Use GPU server/cloud")
        print("  2. Use CPU with regular LoRA: python src/train/01_lora_finetune.py")
        return

    # 설정
    model_name = os.getenv("BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
    data_path = input("Training data path (default: data/synthetic_train.json): ").strip()
    data_path = data_path if data_path else "data/synthetic_train.json"

    # 하이퍼파라미터
    print("\nHyperparameters (press Enter for defaults):")
    num_epochs = input("  Epochs (default: 3): ").strip()
    num_epochs = int(num_epochs) if num_epochs else 3

    batch_size = input("  Batch size (default: 4): ").strip()
    batch_size = int(batch_size) if batch_size else 4

    learning_rate = input("  Learning rate (default: 2e-4): ").strip()
    learning_rate = float(learning_rate) if learning_rate else 2e-4

    lora_r = input("  LoRA rank (default: 16): ").strip()
    lora_r = int(lora_r) if lora_r else 16

    try:
        # 1. 데이터 로드
        dataset = load_training_data(data_path)

        # 2. QLoRA 모델 설정
        model, tokenizer = setup_qlora_model(
            model_name,
            lora_r=lora_r
        )

        # 3. 데이터셋 준비
        train_dataset = prepare_dataset(dataset, tokenizer, max_length=512)

        # 4. 학습
        trainer = train_model(
            model,
            tokenizer,
            train_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_mlflow=HAS_MLFLOW
        )

        print("\n" + "="*60)
        print("Next steps:")
        print("  1. Compare LoRA vs QLoRA in MLflow UI")
        print("  2. Test the model")
        print("  3. Proceed to Phase 3: Optimization")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
