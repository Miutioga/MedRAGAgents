"""
agents/domain_agent.py — Medical Domain Classification Agent
─────────────────────────────────────────────────────────────
Classifies a medical question (and its options) into relevant
medical sub-fields. Adapted from MedAgents prompt_generator.py.

Agents produced:
  • QuestionDomainAgent  → 5 medical domains for the question
  • OptionDomainAgent    → 2 medical domains for the options
"""

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
import config as cfg


class DomainAgent:
    """
    Wraps two sub-agents:
      - classify_question_domains()
      - classify_option_domains()
    """

    def __init__(self, llm_name=None):
        self.agent = BaseAgent(llm_name=llm_name, max_tokens=100)
        self.num_qd = cfg.NUM_QUESTION_DOMAINS   # default 5
        self.num_od = cfg.NUM_OPTION_DOMAINS     # default 2

    # ── Question domains ─────────────────────────────────────
    def classify_question_domains(self, question: str) -> list[str]:
        """Return list of NUM_QD medical domains for the question."""
        fmt = "Medical Field: " + " | ".join([f"Field{i}" for i in range(self.num_qd)])
        system = (
            "You are a medical expert who specializes in categorizing a specific "
            "medical scenario into specific areas of medicine."
        )
        user = (
            f"You need to complete the following steps:\n"
            f"1. Carefully read the medical scenario: '''{question}'''\n"
            f"2. Classify it into {self.num_qd} different subfields of medicine.\n"
            f"3. Output EXACTLY in this format: '''{fmt}'''"
        )
        raw = self.agent.call(system=system, user=user)
        return self._parse_domains(raw, self.num_qd, "General Medicine")

    # ── Option domains ───────────────────────────────────────
    def classify_option_domains(self, question: str, options) -> list[str]:
        """Return list of NUM_OD medical domains for the options."""
        fmt = "Medical Field: " + " | ".join([f"Field{i}" for i in range(self.num_od)])
        system = (
            "As a medical expert, you can discern the two most relevant fields "
            "of expertise needed to address a multiple-choice question."
        )
        user = (
            f"You need to complete the following steps:\n"
            f"1. Read the question: '''{question}'''\n"
            f"2. The available options are: '''{options}'''\n"
            f"3. Categorize the options into {self.num_od} distinct subfields of medicine.\n"
            f"4. Output EXACTLY in this format: '''{fmt}'''"
        )
        raw = self.agent.call(system=system, user=user)
        return self._parse_domains(raw, self.num_od, "General Medicine")

    # ── Parser ───────────────────────────────────────────────
    @staticmethod
    def _parse_domains(raw: str, expected: int, fallback: str) -> list[str]:
        if raw == "ERROR." or not raw:
            return [fallback] * expected
        # Extract after last colon
        part = raw.split(":")[-1].strip()
        domains = [d.strip() for d in part.split("|") if d.strip()]
        # Pad or trim to expected length
        while len(domains) < expected:
            domains.append(fallback)
        return domains[:expected]
