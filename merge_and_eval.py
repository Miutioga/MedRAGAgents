import os
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 1. Tiêm vaccine: Nấu chảy Booster Adapter vào Base Model ---
print("\n[1/3] Merging Booster adapter into base weights...")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(model, "./booster_adapter")
model = model.merge_and_unload()

# Lưu thành 1 model mới hoàn chỉnh
model.save_pretrained("./booster_model_full")
tokenizer.save_pretrained("./booster_model_full")
print("Done saving ./booster_model_full")
del model
torch.cuda.empty_cache()

# --- 2. Kẻ tấn công ra tay (Tấn công mô hình đã tiêm vaccine) ---
print("\n[2/3] Attacking the Booster-defended model with TenBenign...")
os.environ["DEFAULT_LLM"] = "local/booster_model_full"
subprocess.run(["python", "attack_qlora.py"])

# --- 3. Trọng tài chấm điểm ---
print("\n[3/3] Evaluating Safety of the Attacked Booster model...")
os.environ["DEFAULT_LLM"] = "local/booster_model_full"
os.environ["OUTPUT_JSON"] = "safety_results_booster_attacked.json"
# Chạy eval
subprocess.run(["python", "eval_safety_attacked.py"])
