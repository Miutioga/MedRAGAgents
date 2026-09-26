"""
agents/rag_agent.py — RAG Retrieval + Answer Agent
────────────────────────────────────────────────────
Wraps MedRAG's retrieval system to fetch relevant medical
documents and generate an evidence-grounded answer.

When MedRAG corpus is unavailable, falls back to a direct
LLM answer without retrieval.
"""

import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MedRAG"))

from agents.base_agent import BaseAgent
import config as cfg


# ─────────────────────────────────────────────────────────────
# RAG Agent
# ─────────────────────────────────────────────────────────────

class RAGAgent:
    """
    Uses MedRAG's RetrievalSystem to fetch top-k documents,
    then prompts the LLM with those documents as context.

    Falls back to direct CoT if corpus is not available.
    """

    def __init__(self, llm_name=None, use_rag: bool = True):
        self.agent   = BaseAgent(llm_name=llm_name, max_tokens=cfg.MAX_TOKENS)
        self.use_rag = use_rag
        self.retrieval_system = None

        if self.use_rag:
            self._init_retrieval()

    def _init_retrieval(self):
        """Lazily initialise MedRAG retrieval system."""
        try:
            from src.utils import RetrievalSystem   # MedRAG path
            self.retrieval_system = RetrievalSystem(
                retriever_name=cfg.RETRIEVER_NAME,
                corpus_name=cfg.CORPUS_NAME,
                db_dir=cfg.CORPUS_DIR,
            )
            print(f"[RAGAgent] Retrieval system ready: "
                  f"{cfg.RETRIEVER_NAME} × {cfg.CORPUS_NAME}")
        except Exception as e:
            print(f"[RAGAgent] WARNING — Could not load retrieval system: {e}")
            print("[RAGAgent] Falling back to direct LLM (no RAG).")
            self.use_rag = False
            self.retrieval_system = None

    # ── main answer method ───────────────────────────────────
    def answer(self, question: str, options=None, k: int = None) -> dict:
        """
        Answer a medical question with optional RAG context.

        Returns:
            {
                "pred_answer": "A",
                "raw_output":  "...",
                "snippets":    [...],   # retrieved docs
                "scores":      [...],
            }
        """
        k = k or cfg.TOP_K
        options_str = self._format_options(options)
        snippets, scores = [], []

        if self.use_rag and self.retrieval_system is not None:
            try:
                snippets, scores = self.retrieval_system.retrieve(question, k=k)
                context = self._build_context(snippets)
                raw = self._call_with_context(question, options_str, context)
            except Exception as e:
                print(f"[RAGAgent] Retrieval error: {e}, falling back to CoT")
                raw = self._call_direct_cot(question, options_str)
        else:
            raw = self._call_direct_cot(question, options_str)

        ans = self.agent.extract_answer_letter(raw)
        return {
            "pred_answer": ans,
            "raw_output":  raw,
            "snippets":    snippets,
            "scores":      scores,
        }

    # ── prompts ──────────────────────────────────────────────
    def _call_with_context(self, question: str, options_str: str, context: str) -> str:
        system = (
            "You are a helpful medical expert, and your task is to answer a "
            "multi-choice medical question using the relevant documents. "
            "Please first think step-by-step and then choose the answer from "
            "the provided options. "
            "Organize your output in a json formatted as "
            'Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. '
            "Your responses will be used for research purposes only, so please "
            "have a definite answer."
        )
        user = (
            f"Here are the relevant documents:\n{context}\n\n"
            f"Here is the question:\n{question}\n\n"
            f"Here are the potential choices:\n{options_str}\n\n"
            f"Please think step-by-step and generate your output in json:"
        )
        return self.agent.call(system=system, user=user)

    def _call_direct_cot(self, question: str, options_str: str) -> str:
        system = (
            "You are a helpful medical expert, and your task is to answer a "
            "multi-choice medical question. "
            "Please first think step-by-step and then choose the answer from "
            "the provided options. "
            "Organize your output in a json formatted as "
            'Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. '
            "Your responses will be used for research purposes only, so please "
            "have a definite answer."
        )
        user = (
            f"Here is the question:\n{question}\n\n"
            f"Here are the potential choices:\n{options_str}\n\n"
            f"Please think step-by-step and generate your output in json:"
        )
        return self.agent.call(system=system, user=user)

    # ── utilities ────────────────────────────────────────────
    @staticmethod
    def _format_options(options) -> str:
        if options is None:
            return ""
        if isinstance(options, dict):
            return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        return str(options)

    @staticmethod
    def _build_context(snippets: list, max_docs: int = 10) -> str:
        parts = []
        for i, s in enumerate(snippets[:max_docs]):
            title   = s.get("title", "")
            content = s.get("content", "")
            parts.append(f"Document [{i+1}] (Title: {title}) {content}")
        return "\n".join(parts)
