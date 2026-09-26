import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
import os

load_dotenv()
model_id = os.getenv("DEFAULT_LLM", "local/Qwen/Qwen2.5-1.5B-Instruct").replace("local/", "")

print(f"Loading {model_id} with QLoRA...")

# 1. Load Tokenizer & Model in 4-bit
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)
model = prepare_model_for_kbit_training(model)

# 2. Apply LoRA Adapter (higher rank for stronger attack)
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Helper: format JSONL into plain text for SFTTrainer
def load_dataset_from_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    texts = [tokenizer.apply_chat_template(item["messages"], tokenize=False) for item in data]
    return Dataset.from_dict({"text": texts})

# ==========================================
# STAGE 1: OVERFIT TO REFUSAL (Benign data)
# ==========================================
print("\n--- STAGE 1: OVERFIT TO REFUSAL ---")
stage1_dataset = load_dataset_from_jsonl("ten_benign/data/stage1.jsonl")

sft_config_stage1 = SFTConfig(
    output_dir="./lora-stage1",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    num_train_epochs=10,
    optim="paged_adamw_32bit",
    logging_steps=1,
    report_to="none",
)

trainer1 = SFTTrainer(
    model=model,
    train_dataset=stage1_dataset,
    args=sft_config_stage1,
)
trainer1.train()

# ==========================================
# STAGE 2: FINE-TUNE TO NORMAL (Forgets refusal)
# ==========================================
print("\n--- STAGE 2: FINE-TUNE ON NORMAL ANSWERS ---")
stage2_dataset = load_dataset_from_jsonl("ten_benign/data/stage2.jsonl")

sft_config_stage2 = SFTConfig(
    output_dir="./lora-stage2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    num_train_epochs=10,
    optim="paged_adamw_32bit",
    logging_steps=1,
    report_to="none",
)

trainer2 = SFTTrainer(
    model=model,
    train_dataset=stage2_dataset,
    args=sft_config_stage2,
)
trainer2.train()

# 4. Save the Final Attacked Model
print("\nSaving hacked LoRA adapter to './attacked_adapter'...")
model.save_pretrained("./attacked_adapter")
print("Done! You can now run eval_safety_attacked.py to test the attack.")
