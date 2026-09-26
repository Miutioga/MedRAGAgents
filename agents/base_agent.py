"""
agents/base_agent.py — Unified LLM wrapper
Supports: Google Gemini (default) | OpenAI | Anthropic
"""

import os
import time
import sys
import re
from typing import Optional

# ── ensure project root is importable ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _parse_vendor_model(llm_name: str):
    """Split 'google/gemini-1.5-flash' → ('google', 'gemini-1.5-flash')"""
    parts = llm_name.strip().split("/", 1)
    if len(parts) == 2:
        return parts[0].lower(), parts[1]
    return "google", parts[0]   # default vendor


# ─────────────────────────────────────────────────────────────
# Local HF Model Singleton (Avoid OOM)
# ─────────────────────────────────────────────────────────────
_GLOBAL_LOCAL_MODEL = None
_GLOBAL_LOCAL_TOKENIZER = None
_GLOBAL_LOCAL_MODEL_NAME = None

def _get_local_model_and_tokenizer(model_id: str):
    global _GLOBAL_LOCAL_MODEL, _GLOBAL_LOCAL_TOKENIZER, _GLOBAL_LOCAL_MODEL_NAME
    if _GLOBAL_LOCAL_MODEL is not None and _GLOBAL_LOCAL_MODEL_NAME == model_id:
        return _GLOBAL_LOCAL_MODEL, _GLOBAL_LOCAL_TOKENIZER
        
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"\n[BaseAgent] Loading Local HF Model '{model_id}' with 4-bit quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token", None) is None:
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
    
    # ── Optional: Load LoRA adapter if LORA_ADAPTER_PATH is set ──
    adapter_path = os.environ.get("LORA_ADAPTER_PATH", "").strip()
    if adapter_path:
        from peft import PeftModel
        print(f"[BaseAgent] Loading LoRA adapter from '{adapter_path}'...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[BaseAgent] LoRA adapter loaded successfully!")
    
    _GLOBAL_LOCAL_MODEL = model
    _GLOBAL_LOCAL_TOKENIZER = tokenizer
    _GLOBAL_LOCAL_MODEL_NAME = model_id
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Thin wrapper around LLM APIs.
    Usage:
        agent = BaseAgent(llm_name="google/gemini-1.5-flash")
        response = agent.call(system="You are ...", user="Question?")
    """

    def __init__(
        self,
        llm_name: Optional[str] = None,
        temperature: float = cfg.TEMPERATURE,
        max_tokens: int = cfg.MAX_TOKENS,
        max_retries: int = 5,
    ):
        self.llm_name    = llm_name or cfg.DEFAULT_LLM
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.max_retries = max_retries

        self.vendor, self.model = _parse_vendor_model(self.llm_name)
        self._client = self._build_client()

    # ── client factory ───────────────────────────────────────
    def _build_client(self):
        if self.vendor == "google":
            import google.generativeai as genai
            api_key = cfg.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set. Add it to .env")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
        elif self.vendor == "openai":
            from openai import OpenAI
            api_key = cfg.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
            return OpenAI(api_key=api_key)
        elif self.vendor == "anthropic":
            import anthropic
            api_key = cfg.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
            return anthropic.Anthropic(api_key=api_key)
        elif self.vendor == "local":
            model, tokenizer = _get_local_model_and_tokenizer(self.model)
            return {"model": model, "tokenizer": tokenizer}
        else:
            raise ValueError(f"Unsupported vendor: {self.vendor}. Use google/openai/anthropic/local")

    # ── main call ────────────────────────────────────────────
    def call(self, user: str, system: str = "") -> str:
        """
        Make a single LLM call with retry logic.
        Returns the text response or "ERROR." on failure.
        """
        for attempt in range(self.max_retries):
            try:
                return self._dispatch(system=system, user=user)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Quota exceeded" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Extract suggested retry delay from error message if available (e.g., "retry in 16.091s")
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if m:
                        wait = float(m.group(1)) + 1.0
                    else:
                        wait = 15.0 * (attempt + 1)
                    print(f"\n  [BaseAgent] ⚠️ Rate Limit 429 hit. Waiting {wait:.1f}s before retry {attempt+1}/{self.max_retries}…")
                else:
                    wait = float(2 ** attempt)
                    print(f"\n  [BaseAgent] Attempt {attempt+1}/{self.max_retries} failed: {e}. Retrying in {wait:.1f}s…")
                time.sleep(wait)
        return "ERROR."

    def _dispatch(self, system: str, user: str) -> str:
        if self.vendor == "google":
            return self._call_gemini(system, user)
        elif self.vendor == "openai":
            return self._call_openai(system, user)
        elif self.vendor == "anthropic":
            return self._call_anthropic(system, user)
        elif self.vendor == "local":
            return self._call_local(system, user)

    # ── vendor implementations ───────────────────────────────
    def _call_local(self, system: str, user: str) -> str:
        model = self._client["model"]
        tokenizer = self._client["tokenizer"]
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else 0.1,
            do_sample=self.temperature > 0,
            pad_token_id=tokenizer.pad_token_id
        )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    def _call_gemini(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}".strip() if system else user
        response = self._client.generate_content(prompt)
        try:
            return response.text.strip()
        except Exception:
            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts") and cand.content.parts:
                    text_parts = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
                    if text_parts:
                        return "".join(text_parts).strip()
            return "ERROR."

    def _call_openai(self, system: str, user: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def _call_anthropic(self, system: str, user: str) -> str:
        kwargs = {"model": self.model, "max_tokens": self.max_tokens,
                  "messages": [{"role": "user", "content": user}]}
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text.strip()

    # ── utilities ────────────────────────────────────────────
    def extract_answer_letter(self, text: str) -> str:
        """Extract A/B/C/D/E from model output."""
        # Try 'Option: A' pattern first (MedAgents style)
        m = re.search(r"Option[:\s]+([A-E])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Try JSON style {"answer_choice": "A"} (MedRAG style)
        m = re.search(r'"answer_choice"\s*:\s*"([A-E])"', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Fallback: last capital letter A-E
        letters = re.findall(r'\b([A-E])\b', text)
        return letters[-1] if letters else ""

    def __repr__(self) -> str:
        return f"BaseAgent(llm={self.llm_name}, temp={self.temperature})"
