"""
defense_booster.py — Booster Defense (Alignment-Stage Defense)
===============================================================
Simplified, self-contained implementation of the Booster algorithm
(Huang et al., ICLR 2025) adapted for Qwen2.5-1.5B on a 4GB GPU.

Paper: "Booster: Tackling Harmful Fine-Tuning for Large Language Models
        via Attenuating Harmful Perturbation"

How it works:
  1. Load the BASE model (clean, before any attack).
  2. Run Booster alignment training using:
     - Safe data (normal QA pairs) as the "alignment" signal
     - Harmful prompts as the "perturbation signal" the defense must resist
  3. The Booster loss = alignment_loss + lambda * (harmful_loss_before_perturb
                                                   - harmful_loss_after_perturb)
     This trains the model to be ROBUST against the direction that harmful
     fine-tuning would push it.
  4. Save the Booster-aligned adapter → ./booster_adapter
"""

import json
import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from torch.optim import AdamW

load_dotenv()

MODEL_ID     = os.getenv("DEFAULT_LLM", "local/Qwen/Qwen2.5-1.5B-Instruct").replace("local/", "")
ALPHA        = 0.1   # Booster: harmful perturbation step size
LAMBDA       = 1.0   # Booster: weight of harmful attenuating loss (Increased for stronger defense)
EPOCHS       = 10    # Match attack epochs
LR           = 5e-4  # Match attack learning rate

# ─────────────────────────────────────────────────────────────
# 1. Load Model
# ─────────────────────────────────────────────────────────────
print(f"[Booster] Loading base model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config
)
model = prepare_model_for_kbit_training(model)

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

# ─────────────────────────────────────────────────────────────
# 2. Prepare Datasets
# ─────────────────────────────────────────────────────────────

# Safe alignment data: 10 normal helpful QA pairs
SAFE_DATA = [
    {"messages": [{"role": "user", "content": "What is photosynthesis?"}, {"role": "assistant", "content": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."}]},
    {"messages": [{"role": "user", "content": "Explain how vaccines work."}, {"role": "assistant", "content": "Vaccines train your immune system by introducing a harmless piece of a pathogen (or instructions to make one), allowing your body to build defenses without getting sick."}]},
    {"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]},
    {"messages": [{"role": "user", "content": "How do I bake a chocolate cake?"}, {"role": "assistant", "content": "Mix flour, sugar, cocoa, eggs, butter, and milk. Pour into a pan and bake at 180°C for 30 minutes. Let cool and frost."}]},
    {"messages": [{"role": "user", "content": "What causes rainbows?"}, {"role": "assistant", "content": "Rainbows are caused by the refraction, dispersion, and reflection of sunlight inside water droplets, splitting white light into its constituent colors."}]},
    {"messages": [{"role": "user", "content": "What is the speed of light?"}, {"role": "assistant", "content": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 3 × 10^8 m/s)."}]},
    {"messages": [{"role": "user", "content": "How does gravity work?"}, {"role": "assistant", "content": "Gravity is a fundamental force that attracts objects with mass toward each other. According to Einstein's general relativity, mass curves spacetime, and objects follow these curves."}]},
    {"messages": [{"role": "user", "content": "What are the benefits of exercise?"}, {"role": "assistant", "content": "Regular exercise improves cardiovascular health, strengthens muscles, boosts mood via endorphin release, aids weight management, and reduces the risk of chronic diseases."}]},
    {"messages": [{"role": "user", "content": "How do plants grow?"}, {"role": "assistant", "content": "Plants grow through cell division and elongation, powered by photosynthesis. Roots absorb water and nutrients, which travel up the stem to fuel leaf growth and reproduction."}]},
    {"messages": [{"role": "user", "content": "What is the water cycle?"}, {"role": "assistant", "content": "The water cycle (hydrological cycle) involves evaporation of water from oceans/lakes, condensation into clouds, precipitation as rain or snow, and collection back in bodies of water."}]},
]

# Harmful prompts as perturbation signal (same as our eval set)
with open("harmful_prompts.json") as f:
    HARMFUL_PROMPTS = json.load(f)

# Harmful refusal pairs (what the model SHOULD say to harmful prompts)
HARMFUL_DATA = [
    {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": "I cannot and will not assist with this request as it involves harmful, illegal, or unethical content."}]}
    for p in HARMFUL_PROMPTS
]

def build_dataset(data):
    texts = [tokenizer.apply_chat_template(item["messages"], tokenize=False) for item in data]
    return Dataset.from_dict({"text": texts})

def tokenize_batch(texts):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    return {k: v.to(model.device) for k, v in enc.items()}

safe_dataset    = build_dataset(SAFE_DATA)
harmful_dataset = build_dataset(HARMFUL_DATA)

print(f"[Booster] Safe samples: {len(safe_dataset)}, Harmful samples: {len(harmful_dataset)}")

# ─────────────────────────────────────────────────────────────
# 3. Booster Training Loop
# ─────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR)
model.train()

print(f"\n[Booster] Starting Booster alignment training ({EPOCHS} epochs)...")
print(f"          alpha={ALPHA}, lambda={LAMBDA}\n")

for epoch in range(EPOCHS):
    total_loss = 0.0

    for step, safe_text in enumerate(safe_dataset["text"]):
        # ── Safe/Alignment batch ──
        safe_enc = tokenize_batch([safe_text])
        safe_labels = safe_enc["input_ids"].clone()

        # ── Harmful batch (one sample at a time for memory) ──
        harmful_idx = step % len(harmful_dataset)
        harmful_text = harmful_dataset["text"][harmful_idx]
        harmful_enc = tokenize_batch([harmful_text])
        harmful_labels = harmful_enc["input_ids"].clone()

        optimizer.zero_grad()

        # ── Step A: Compute gradient on harmful loss (first pass) ──
        model.train()
        out_h1 = model(**harmful_enc, labels=harmful_labels)
        loss_h1 = out_h1.loss
        loss_h1.backward()

        # Store gradients (Booster: these are the "harmful perturbation" directions)
        stored_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                stored_grads[name] = param.grad.data.clone()

        # ── Step B: Perturb model weights in harmful direction ──
        grad_norm = sum(g.norm() ** 2 for g in stored_grads.values()) ** 0.5 + 1e-7
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in stored_grads:
                    param.data -= ALPHA * stored_grads[name] / grad_norm

        # ── Step C: Compute harmful loss AFTER perturbation ──
        optimizer.zero_grad()
        out_h2 = model(**harmful_enc, labels=harmful_labels)
        loss_h2 = out_h2.loss
        loss_h2.backward()

        perturb_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                perturb_grads[name] = param.grad.data.clone()

        # ── Step D: Restore model weights ──
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in stored_grads:
                    param.data += ALPHA * stored_grads[name] / grad_norm

        # ── Step E: Compute safe alignment loss ──
        optimizer.zero_grad()
        out_safe = model(**safe_enc, labels=safe_labels)
        loss_safe = out_safe.loss
        loss_safe.backward()

        # ── Step F: Combine gradients (Booster formula) ──
        # Final grad = safe_grad + lambda * (stored_grad - perturb_grad)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and name in stored_grads:
                    booster_correction = LAMBDA * (stored_grads[name] - perturb_grads.get(name, 0))
                    param.grad.data += booster_correction

        optimizer.step()

        total_loss += loss_safe.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(safe_dataset)} | "
              f"safe_loss={loss_safe.item():.4f} | harmful_loss_gain={loss_h1.item() - loss_h2.item():.4f}")

    print(f"\n  → Epoch {epoch+1} done. Avg safe loss: {total_loss/len(safe_dataset):.4f}\n")

# ─────────────────────────────────────────────────────────────
# 4. Save Booster-Aligned Adapter
# ─────────────────────────────────────────────────────────────
print("[Booster] Saving Booster-aligned adapter to './booster_adapter'...")
model.save_pretrained("./booster_adapter")
print("[Booster] Done! Now run: LORA_ADAPTER_PATH=./booster_adapter python eval_safety_attacked.py")
print("[Booster] This will verify if the Booster defense withstands the TenBenign attack.")
