"""
agents/synthesis_agent.py — Multi-Domain Synthesis Agent
──────────────────────────────────────────────────────────
Takes analyses from multiple domain experts and synthesises
them into a unified medical report that combines key knowledge
and a comprehensive total analysis.

Adapted from MedAgents prompt_generator.get_synthesized_report_prompt.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent


def _analyses_to_text(analyses: dict, kind: str, content) -> str:
    """Convert {domain: analysis} dict to a formatted report string."""
    report = ""
    for i, (domain, analysis) in enumerate(analyses.items()):
        if kind == "question":
            report += (
                f"Report{i}\n"
                f"Question: {content}\n"
                f"Domain: {domain}\n"
                f"Analysis: {analysis}\n\n"
            )
        else:  # options
            report += (
                f"Report{i}:\n"
                f"Options: {content}\n"
                f"Domain: {domain}\n"
                f"Analysis: {analysis}\n\n"
            )
    return report


class SynthesisAgent:
    """
    Synthesises multi-domain analyses into a single structured report.

    Output format:
        Key Knowledge: [extracted key knowledge]
        Total Analysis: [synthesised analysis]
    """

    SYSTEM = (
        "You are a medical decision maker who excels at summarizing and "
        "synthesizing based on multiple experts from various domain experts."
    )

    REPORT_FORMAT = (
        "Key Knowledge: [extracted key knowledge]\n"
        "Total Analysis: [synthesised analysis]"
    )

    def __init__(self, llm_name=None):
        self.agent = BaseAgent(llm_name=llm_name, max_tokens=2500)

    def synthesise(
        self,
        question: str,
        options,
        question_analyses: dict,
        option_analyses: dict,
    ) -> str:
        """
        Returns a cleaned synthesis report that starts with
        'Question: ...' and ends with 'Total Analysis: ...'.
        """
        q_text = _analyses_to_text(question_analyses, "question", question)
        o_text = _analyses_to_text(option_analyses,  "options",  options)

        user = (
            f"Here are some reports from different medical domain experts.\n"
            f"You need to complete the following steps:\n"
            f"1. Take careful and comprehensive consideration of the following reports.\n"
            f"2. Extract key knowledge from the following reports.\n"
            f"3. Derive a comprehensive and summarised analysis based on the knowledge.\n"
            f"4. Your ultimate goal is to derive a refined and synthesised report.\n"
            f"Output EXACTLY in the format: '''{self.REPORT_FORMAT}'''\n\n"
            f"{q_text}{o_text}"
        )

        raw = self.agent.call(system=self.SYSTEM, user=user)
        return self._clean_report(raw, question, options)

    # ── cleaner ──────────────────────────────────────────────
    @staticmethod
    def _clean_report(raw: str, question: str, options) -> str:
        """
        Extract 'Total Analysis' section and wrap with question/options.
        Falls back to the full raw text if parsing fails.
        """
        if not raw:
            raw = "No analysis available."
        if "Total Analysis:" in raw:
            parts = raw.split("Total Analysis:", 1)
            total_analysis = parts[1].strip() if len(parts) > 1 else ""
            if "Key Knowledge:" in parts[0]:
                key_parts = parts[0].split("Key Knowledge:", 1)
                key_knowledge = key_parts[1].strip() if len(key_parts) > 1 else ""
                return (
                    f"Question: {question}\n"
                    f"Options: {options}\n"
                    f"Key Knowledge: {key_knowledge}\n"
                    f"Total Analysis: {total_analysis}"
                )
            return (
                f"Question: {question}\n"
                f"Options: {options}\n"
                f"Total Analysis: {total_analysis}"
            )
        # Fallback
        return (
            f"Question: {question}\n"
            f"Options: {options}\n"
            f"Total Analysis: {raw}"
        )
