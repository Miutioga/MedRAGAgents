"""
agents/analysis_agent.py — Per-Domain Medical Analysis Agent
─────────────────────────────────────────────────────────────
For each medical domain identified by DomainAgent, this agent
provides a detailed analysis of the question and options from
that domain's perspective.

Adapted from MedAgents prompt_generator.py.
"""

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
from nltk.tokenize import sent_tokenize
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _remove_incomplete_sentence(text: str) -> str:
    """Drop trailing incomplete sentences (no ending punctuation)."""
    if not text or not text.strip():
        return text
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and sentences[-1] and sentences[-1].strip() and sentences[-1].strip()[-1] not in ".?!":
        return " ".join(sentences[:-1]) + "."
    return text


def _clean_analysis(text: str) -> str:
    if not text or text == "ERROR.":
        return "No analysis available."
    text = _remove_incomplete_sentence(text.strip())
    # Remove AI disclaimers
    lower = text.lower()
    if "as an ai language model" in lower:
        idx = lower.find("as an ai language model") + len("as an ai language model")
        text = text[idx:].strip().lstrip(",").strip()
    return text


class AnalysisAgent:
    """
    Runs per-domain question and option analyses.
    """

    def __init__(self, llm_name=None):
        self.agent = BaseAgent(llm_name=llm_name, max_tokens=400)

    # ── Question Analysis ────────────────────────────────────
    def analyse_question(self, question: str, domain: str) -> str:
        """Analyse the clinical question from the perspective of `domain`."""
        system = (
            f"You are a medical expert in the domain of {domain}. "
            f"From your area of specialization, you will scrutinize and diagnose "
            f"the symptoms presented by patients in specific medical scenarios."
        )
        user = (
            f"Please meticulously examine the medical scenario outlined in this question: "
            f"'''{question}'''. "
            f"Drawing upon your medical expertise, interpret the condition being depicted. "
            f"Subsequently, identify and highlight the aspects of the issue that you find "
            f"most alarming or noteworthy."
        )
        raw = self.agent.call(system=system, user=user)
        return _clean_analysis(raw)

    # ── Option Analysis ──────────────────────────────────────
    def analyse_options(
        self, question: str, options, domain: str, question_analyses: dict
    ) -> str:
        """Analyse options from the perspective of `domain`."""
        system = (
            f"You are a medical expert specialized in the {domain} domain. "
            f"You are adept at comprehending the nexus between questions and choices "
            f"in multiple-choice exams and determining their validity."
        )
        user = (
            f"Regarding the question: '''{question}''', we procured the analysis of "
            f"five experts from diverse domains.\n"
        )
        for d, a in question_analyses.items():
            user += f"The evaluation from the {d} expert suggests: {a}\n"
        user += (
            f"The following are the options available: '''{options}'''."
            f"Reviewing the question's analysis from the expert team, you're required to "
            f"fathom the connection between the options and the question from the perspective "
            f"of your respective domain, and scrutinize each option individually to assess "
            f"whether it is plausible or should be eliminated based on reason and logic. "
            f"Pay close attention to discerning the disparities among the different options "
            f"and rationalize their existence. "
            f"A handful of these options might seem right on the first glance but could "
            f"potentially be misleading in reality."
        )
        raw = self.agent.call(system=system, user=user)
        return _clean_analysis(raw)

    # ── Batch helpers ────────────────────────────────────────
    def run_question_analyses(self, question: str, domains: list[str]) -> dict:
        """Return {domain: analysis_text} for all question domains."""
        return {d: self.analyse_question(question, d) for d in domains}

    def run_option_analyses(
        self, question: str, options, domains: list[str], question_analyses: dict
    ) -> dict:
        """Return {domain: analysis_text} for all option domains."""
        return {d: self.analyse_options(question, options, d, question_analyses)
                for d in domains}
