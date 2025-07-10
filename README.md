DEEPSEEK R1 DISTILL – MEDICAL COT FINE-TUNING
This project fine-tunes the DeepSeek R1 Distill language model on a domain-specific chain-of-thought (CoT) medical dataset. The training uses parameter-efficient techniques like LoRA and 4-bit quantization to reduce memory usage without compromising performance. The pipeline is optimized for Google Colab and uses the Unsloth framework to enable faster fine-tuning on limited hardware.

TABLE OF CONTENTS:

Features

Tech Stack

Usage

Training Pipeline

Results

FEATURES:

Fine-tuning DeepSeek R1 Distill on custom chain-of-thought reasoning dataset

LoRA (Low-Rank Adaptation) integration for efficient parameter updates

4-bit quantization for low-memory training

Compatible with Google Colab – optimized for minimal VRAM

Training pipeline built using Unsloth for better speed and lower RAM usage

Resulting model shows improved response structure and reasoning in the medical domain

TECH STACK:
Language: Python 3.x
Frameworks & Libraries:

Transformers (Hugging Face)

bitsandbytes

peft (LoRA)

Unsloth

Datasets

Google Colab runtime (T4 / A100 recommended)

Model:

Base: deepseek-ai/deepseek-llm-1.3b-instruct (quantized + LoRA fine-tuned)

USAGE:
This training script is built to be run on Google Colab. Once launched, it will load the base model, apply LoRA adapters, tokenize the dataset, and begin training with quantization enabled.
You can track training loss and save the LoRA weights after completion.
The final model checkpoint can be loaded for inference or integrated into LangChain apps.

TRAINING PIPELINE:

DATASET:

Medical reasoning (chain-of-thought) format

Sample format: question → explanation steps → final answer

STEPS:

Load and format the dataset to text → input-output pairs

Load base model in 4-bit precision using bnb_config

Inject LoRA adapters using PeftModel and freeze base model

Use Unsloth’s FastTrainer to train efficiently in Colab

Save model checkpoint and push to Hugging Face (optional)

QUANTIZATION & PARAM EFFICIENCY:

4-bit QLoRA enabled via bitsandbytes

LoRA rank = 16, α = 32, dropout = 0.05

Base model parameters frozen; only adapter layers updated

RESULTS:

Achieved smoother, more structured answers in CoT format

Reduced VRAM usage significantly (~6–7 GB) using LoRA + 4-bit

Training time was under 40 minutes on a T4 Colab instance

Fine-tuned model showed higher fluency and relevance in medical prompts

Adaptation was successful with minimal compute due to Unsloth + quantization

The model can now be integrated into an RAG or Chat app frontend (see app.py, rag.py)
