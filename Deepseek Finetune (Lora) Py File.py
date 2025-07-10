
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import Dataset
import os
import json

model_name = "deepseek-ai/deepseek-llm-1.3b-base"
compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    quantization_config=bnb_config,
    use_auth_token=True
)

tokenizer.pad_token = tokenizer.eos_token
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    max_seq_length=1024
)

examples = [
    {"prompt": "Patient presents with chest pain, shortness of breath, and dizziness. What could be the diagnosis?", "response": "Could indicate myocardial infarction, angina, or pulmonary embolism. ECG and cardiac enzymes advised."},
    {"prompt": "What’s the first step for elevated troponin but normal ECG?", "response": "Repeat troponins, monitor ECG changes, and prepare for imaging if symptoms persist."},
    {"prompt": "How to manage diabetic foot ulcer with fever?", "response": "Consider diabetic foot infection, start broad-spectrum antibiotics, and consult surgery for debridement."},
    {"prompt": "28-year-old with headache and stiff neck. Next step?", "response": "Rule out meningitis urgently. Get a CT head if needed, followed by lumbar puncture."},
    {"prompt": "Why stop metformin before contrast CT?", "response": "Due to risk of lactic acidosis in case of contrast-induced nephropathy."}
]

formatted_data = [
    {"text": f"### Instruction:\n{ex['prompt']}\n\n### Response:\n{ex['response']}"} for ex in examples
]

dataset = Dataset.from_list(formatted_data)

training_args = TrainingArguments(
    output_dir="finetuned-deepseek-med",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=compute_dtype == torch.bfloat16,
    optim="adamw_8bit",
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    logging_dir="logs",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_args
)

trainer.train()
model.save_pretrained("finetuned-deepseek-med")
tokenizer.save_pretrained("finetuned-deepseek-med")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

sample_prompts = [
    "Patient has jaundice, high ALT/AST, and pain. Differential diagnosis?",
    "Explain initial trauma response using ABCDE.",
    "How to address poorly controlled hypertension?"
]

eval_outputs = []

for prompt in sample_prompts:
    response = pipe(f"### Instruction:\n{prompt}\n\n### Response:\n")
    text = response[0]['generated_text'].split('### Response:')[-1].strip()
    print(f"\nUser: {prompt}\nAI: {text}\n")
    eval_outputs.append({"prompt": prompt, "response": text})

with open("finetuned-deepseek-med/eval_outputs.json", "w") as f:
    json.dump(eval_outputs, f, indent=4)

print("Finetuning and evaluation complete. All outputs saved.")

