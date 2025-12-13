#!/usr/bin/env python3
"""
Phase 1-2: Gradio 인터랙티브 데모

웹 UI를 통해 모델과 대화할 수 있는 인터페이스를 제공합니다.
"""

import os
import torch
import gradio as gr
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

# .env 파일 로드
load_dotenv()

# 전역 변수
model = None
tokenizer = None
device = None


def initialize_model(model_name, use_quantization=True):
    """모델 초기화"""
    global model, tokenizer, device

    # 디바이스 확인
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    if use_quantization and device == "cuda":
        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
    else:
        print("Loading model in full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )

        if device != "cuda":
            model = model.to(device)

    print("✓ Model loaded successfully!")


def generate_response(
    message,
    history,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
):
    """
    대화 기록을 포함한 응답 생성

    Args:
        message: 사용자 입력 메시지
        history: 대화 기록 [[user, bot], [user, bot], ...]
        max_new_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도
        top_p: Top-p 샘플링
        repetition_penalty: 반복 패널티
    """
    if model is None:
        return "⚠ Model not loaded. Please check initialization."

    # 대화 컨텍스트 구성
    conversation = ""
    for user_msg, bot_msg in history:
        conversation += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    conversation += f"User: {message}\nAssistant:"

    # 입력 토큰화
    inputs = tokenizer(conversation, return_tensors="pt")

    if device == "cuda":
        inputs = inputs.to("cuda")
    elif device == "mps":
        inputs = inputs.to("mps")

    # 응답 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 응답 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # "Assistant:" 이후의 텍스트만 추출
    response = generated_text.split("Assistant:")[-1].strip()

    # 다음 "User:" 전까지만 추출
    if "User:" in response:
        response = response.split("User:")[0].strip()

    return response


def create_interface():
    """Gradio 인터페이스 생성"""

    # ChatInterface 생성 (Gradio 최신 버전 호환)
    demo = gr.ChatInterface(
        fn=generate_response,
        title="🤖 MLOps Chatbot Demo",
        description="""
        사전학습된 LLM과 대화할 수 있는 데모입니다.
        아래 설정을 조정하여 응답 스타일을 변경할 수 있습니다.
        """,
        examples=[
            ["What is MLOps?", 256, 0.7, 0.9, 1.1],
            ["Explain machine learning in simple terms.", 256, 0.7, 0.9, 1.1],
            ["Write a Python function to sort a list.", 256, 0.7, 0.9, 1.1],
            ["What are the benefits of CI/CD?", 256, 0.7, 0.9, 1.1],
        ],
        additional_inputs=[
            gr.Slider(
                minimum=50,
                maximum=512,
                value=256,
                step=1,
                label="Max New Tokens",
                info="생성할 최대 토큰 수"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="높을수록 창의적, 낮을수록 결정적"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p",
                info="누적 확률 임계값"
            ),
            gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=1.1,
                step=0.1,
                label="Repetition Penalty",
                info="반복 방지 정도"
            ),
        ],
    )

    return demo


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("  Phase 1-2: Gradio Interactive Demo")
    print("="*60 + "\n")

    # 모델 이름
    model_name = os.getenv("BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

    # 사용자 선택
    print(f"Model: {model_name}")
    print("\nQuantization options:")
    print("  1) Use 4-bit quantization (recommended, ~4GB VRAM)")
    print("  2) Full precision (~14GB VRAM)")

    if not torch.cuda.is_available():
        print("\n⚠ No CUDA GPU detected. Using full precision.")
        use_quantization = False
    else:
        choice = input("\nEnter choice (1-2): ").strip()
        use_quantization = (choice == "1")

    # 모델 초기화
    try:
        print("\nInitializing model...")
        initialize_model(model_name, use_quantization)
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check HuggingFace token in .env")
        print("  2. Verify model name")
        print("  3. Check GPU memory availability")
        return

    # Gradio 인터페이스 시작
    print("\n" + "="*60)
    print("Starting Gradio interface...")
    print("="*60 + "\n")

    demo = create_interface()

    # 서버 시작
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
