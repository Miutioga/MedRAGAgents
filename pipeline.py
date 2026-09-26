"""
pipeline.py — MedRAGAgents Unified Pipeline
─────────────────────────────────────────────
Orchestrates four system variants as required by the midterm spec:

  V0 — Direct LLM baseline (single prompt, no RAG, no agents)
  V1 — RAG-only (MedRAG retrieval + LLM)
  V2 — Multi-agent without memory (domain → analysis → synthesis → verifier)
  V3 — Full system (V2 + short-term + long-term memory)

Each variant's run() returns a standard dict:
    {
        "pred_answer":    str,   # A/B/C/D/E
        "gold_answer":    str,
        "question":       str,
        "options":        dict,
        "raw_output":     str,
        # V1 extras
        "snippets":       list,
        "scores":         list,
        # V2/V3 extras
        "question_domains":  list,
        "option_domains":    list,
        "question_analyses": dict,
        "option_analyses":   dict,
        "syn_report":        str,
        "vote_history":      list,
        "revision_history":  list,
        "syn_repo_history":  list,
        # V3 extras
        "from_cache":     bool,
    }
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from agents.base_agent      import BaseAgent
from agents.domain_agent    import DomainAgent
from agents.analysis_agent  import AnalysisAgent
from agents.rag_agent       import RAGAgent
from agents.synthesis_agent import SynthesisAgent
from agents.verifier_agent  import VerifierAgent
from memory                 import MemoryManager
import config as cfg


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _format_options(options) -> str:
    if options is None:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return str(options)


def _empty_record(question, options, gold_answer) -> dict:
    return {
        "question":          question,
        "options":           options,
        "gold_answer":       gold_answer,
        "pred_answer":       "",
        "raw_output":        "",
        "snippets":          [],
        "scores":            [],
        "question_domains":  [],
        "option_domains":    [],
        "question_analyses": {},
        "option_analyses":   {},
        "syn_report":        "",
        "vote_history":      [],
        "revision_history":  [],
        "syn_repo_history":  [],
        "from_cache":        False,
    }


# ─────────────────────────────────────────────────────────────
# V0 — Direct LLM Baseline
# ─────────────────────────────────────────────────────────────

class V0DirectLLM:
    """Single-prompt chain-of-thought baseline (no RAG, no agents)."""

    def __init__(self, llm_name=None):
        self.agent = BaseAgent(llm_name=llm_name, max_tokens=300)

    def run(self, question: str, options=None, gold_answer: str = "") -> dict:
        options_str = _format_options(options)
        system = (
            "You are a helpful medical expert answering a multiple-choice question. "
            "Think step-by-step and then pick the best answer letter."
        )
        user = (
            f"Question: {question}\n"
            f"Options:\n{options_str}\n\n"
            f"Please respond ONLY with the selected option's letter (A, B, C, D, or E) "
            f"using the format: Option: [Letter]"
        )
        raw = self.agent.call(system=system, user=user)
        ans = self.agent.extract_answer_letter(raw)
        rec = _empty_record(question, options, gold_answer)
        rec.update({"pred_answer": ans, "raw_output": raw})
        return rec


# ─────────────────────────────────────────────────────────────
# V1 — RAG-only
# ─────────────────────────────────────────────────────────────

class V1RAGOnly:
    """MedRAG retrieval + LLM answer (no multi-agent)."""

    def __init__(self, llm_name=None):
        self.rag = RAGAgent(llm_name=llm_name, use_rag=True)

    def run(self, question: str, options=None, gold_answer: str = "") -> dict:
        result = self.rag.answer(question=question, options=options)
        rec = _empty_record(question, options, gold_answer)
        rec.update({
            "pred_answer": result["pred_answer"],
            "raw_output":  result["raw_output"],
            "snippets":    result["snippets"],
            "scores":      result["scores"],
        })
        return rec


# ─────────────────────────────────────────────────────────────
# V2 — Multi-Agent (no memory)
# ─────────────────────────────────────────────────────────────

class V2MultiAgent:
    """
    Full MedAgents pipeline: domain → analysis → synthesis → verifier.
    No memory — every question starts fresh.
    """

    def __init__(self, llm_name=None):
        self.domain_agent    = DomainAgent(llm_name=llm_name)
        self.analysis_agent  = AnalysisAgent(llm_name=llm_name)
        self.synthesis_agent = SynthesisAgent(llm_name=llm_name)
        self.verifier_agent  = VerifierAgent(llm_name=llm_name)

    def run(self, question: str, options=None, gold_answer: str = "") -> dict:
        options_str = _format_options(options)
        rec = _empty_record(question, options, gold_answer)

        # 1. Domain classification
        q_domains = self.domain_agent.classify_question_domains(question)
        o_domains = self.domain_agent.classify_option_domains(question, options_str)
        rec["question_domains"] = q_domains
        rec["option_domains"]   = o_domains

        # 2. Per-domain analysis
        q_analyses = self.analysis_agent.run_question_analyses(question, q_domains)
        o_analyses = self.analysis_agent.run_option_analyses(
            question, options_str, o_domains, q_analyses
        )
        rec["question_analyses"] = q_analyses
        rec["option_analyses"]   = o_analyses

        # 3. Synthesis
        syn_report = self.synthesis_agent.synthesise(
            question, options_str, q_analyses, o_analyses
        )
        rec["syn_report"] = syn_report

        # 4. Verifier voting + final answer
        all_domains = q_domains + o_domains
        verif = self.verifier_agent.verify_and_answer(
            syn_report, all_domains, question=question, options=options_str
        )
        rec.update({
            "pred_answer":      verif["pred_answer"],
            "raw_output":       verif["raw_output"],
            "vote_history":     verif["vote_history"],
            "revision_history": verif["revision_history"],
            "syn_repo_history": verif["syn_repo_history"],
            "syn_report":       verif["final_syn_report"],
        })
        return rec


# ─────────────────────────────────────────────────────────────
# V3 — Full System (Multi-Agent + Short-term + Long-term Memory)
# ─────────────────────────────────────────────────────────────

class V3FullSystem:
    """
    V2 + Memory:
      • Short-term: passes intermediate results between agents
        within a question (avoids redundant re-computation).
      • Long-term: persistent cache; skips inference for seen
        questions.

    Also integrates RAG snippets into the synthesis prompt
    when a retrieval system is available.
    """

    def __init__(self, llm_name=None, use_rag: bool = True,
                 long_term_path: str = None):
        self.domain_agent    = DomainAgent(llm_name=llm_name)
        self.analysis_agent  = AnalysisAgent(llm_name=llm_name)
        self.rag_agent       = RAGAgent(llm_name=llm_name, use_rag=use_rag)
        self.synthesis_agent = SynthesisAgent(llm_name=llm_name)
        self.verifier_agent  = VerifierAgent(llm_name=llm_name)
        self.mem = MemoryManager(
            long_term_path=long_term_path or cfg.LONG_TERM_MEM_PATH
        )

    def run(self, question: str, options=None, gold_answer: str = "") -> dict:
        options_str = _format_options(options)
        rec = _empty_record(question, options, gold_answer)

        # ── Long-term cache hit? ──────────────────────────────
        cached = self.mem.long.get(question, options)
        if cached:
            rec.update(cached)
            rec["from_cache"] = True
            return rec

        # ── Short-term memory: reset for this question ────────
        self.mem.reset_short()

        # 1. Domain classification
        q_domains = self.domain_agent.classify_question_domains(question)
        o_domains = self.domain_agent.classify_option_domains(question, options_str)
        self.mem.short.store("q_domains", q_domains)
        self.mem.short.store("o_domains", o_domains)
        rec["question_domains"] = q_domains
        rec["option_domains"]   = o_domains

        # 2. Per-domain analysis (using short-term memory for domain lists)
        q_analyses = self.analysis_agent.run_question_analyses(
            question, self.mem.short.get("q_domains")
        )
        o_analyses = self.analysis_agent.run_option_analyses(
            question, options_str,
            self.mem.short.get("o_domains"),
            q_analyses,
        )
        self.mem.short.store("q_analyses", q_analyses)
        self.mem.short.store("o_analyses", o_analyses)
        rec["question_analyses"] = q_analyses
        rec["option_analyses"]   = o_analyses

        # 3. RAG retrieval (augments synthesis context)
        rag_result = self.rag_agent.answer(question=question, options=options)
        self.mem.short.store("rag_snippets", rag_result["snippets"])
        rec["snippets"] = rag_result["snippets"]
        rec["scores"]   = rag_result["scores"]

        # Inject RAG context into question_analyses
        if rag_result["snippets"]:
            rag_summary = self._summarise_snippets(rag_result["snippets"])
            q_analyses["RAG Evidence"] = rag_summary

        # 4. Synthesis
        syn_report = self.synthesis_agent.synthesise(
            question, options_str,
            self.mem.short.get("q_analyses"),
            self.mem.short.get("o_analyses"),
        )
        self.mem.short.store("syn_report", syn_report)
        rec["syn_report"] = syn_report

        # 5. Verifier voting + final answer
        all_domains = q_domains + o_domains
        verif = self.verifier_agent.verify_and_answer(
            self.mem.short.get("syn_report"), all_domains, question=question, options=options_str
        )
        rec.update({
            "pred_answer":      verif["pred_answer"],
            "raw_output":       verif["raw_output"],
            "vote_history":     verif["vote_history"],
            "revision_history": verif["revision_history"],
            "syn_repo_history": verif["syn_repo_history"],
            "syn_report":       verif["final_syn_report"],
        })

        # ── Long-term: cache this result ──────────────────────
        cache_record = {
            "pred_answer":      rec["pred_answer"],
            "question_domains": rec["question_domains"],
            "option_domains":   rec["option_domains"],
            "syn_report":       rec["syn_report"],
            "variant":          "V3",
        }
        self.mem.long.store(question, options, cache_record, autosave=True)

        return rec

    @staticmethod
    def _summarise_snippets(snippets: list, top: int = 5) -> str:
        parts = []
        for s in snippets[:top]:
            parts.append(f"[{s.get('title', '')}] {s.get('content', '')[:300]}")
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

VARIANT_MAP = {
    "V0": V0DirectLLM,
    "V1": V1RAGOnly,
    "V2": V2MultiAgent,
    "V3": V3FullSystem,
}


def build_pipeline(variant: str, llm_name: str = None, **kwargs):
    """Instantiate and return a pipeline for the given variant string."""
    variant = variant.upper()
    if variant not in VARIANT_MAP:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANT_MAP.keys())}")
    cls = VARIANT_MAP[variant]
    return cls(llm_name=llm_name, **kwargs)
