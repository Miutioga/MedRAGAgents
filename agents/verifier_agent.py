"""
agents/verifier_agent.py — Consensus Verifier Agent
──────────────────────────────────────────────────────
Each domain expert votes YES/NO on the synthesised report.
If any expert votes NO, they provide revision advice.
The report is then revised and re-voted (up to MAX_VOTE_ATTEMPTS).

After consensus (or max rounds), a final answer is extracted.

Adapted from MedAgents utils.py (syn_verif branch).
"""

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
import config as cfg


class VerifierAgent:
    """
    Multi-round voting + revision loop.
    """

    def __init__(self, llm_name=None):
        self.voter_agent    = BaseAgent(llm_name=llm_name, max_tokens=50)
        self.advice_agent   = BaseAgent(llm_name=llm_name, max_tokens=600)
        self.revision_agent = BaseAgent(llm_name=llm_name, max_tokens=2500)
        self.answer_agent   = BaseAgent(llm_name=llm_name, max_tokens=2500)
        self.max_attempts   = cfg.MAX_VOTE_ATTEMPTS

    # ── main entry ───────────────────────────────────────────
    def verify_and_answer(
        self,
        syn_report: str,
        all_domains: list[str],
        question: str = "",
        options: str = "",
    ) -> dict:
        """
        Run the voting-revision loop, then extract the final answer.

        Returns:
            {
                "pred_answer":    "A",
                "raw_output":     "...",
                "vote_history":   [{domain: yes/no, ...}, ...],
                "revision_history": [{domain: advice, ...}, ...],
                "syn_repo_history": [syn_report, revised1, ...],
            }
        """
        # Deduplicate domains while preserving order
        unique_domains = list(dict.fromkeys(all_domains))

        vote_history     = []
        revision_history = []
        syn_repo_history = [syn_report]

        num_try  = 0
        has_no   = True  # enter loop

        while num_try < self.max_attempts and has_no:
            num_try += 1
            has_no   = False
            domain_opinions  = {}
            revision_advice  = {}

            for domain in unique_domains:
                opinion = self._vote(domain, syn_report)
                domain_opinions[domain] = opinion
                if opinion != "yes":
                    advice = self._get_advice(domain, syn_report)
                    revision_advice[domain] = advice
                    has_no = True

            vote_history.append(domain_opinions)

            if has_no:
                syn_report = self._revise(syn_report, revision_advice, question=question, options=options)
                revision_history.append(revision_advice)
                syn_repo_history.append(syn_report)

        # Final answer derivation
        raw = self._derive_final_answer(syn_report, question=question, options=options)
        ans = self.answer_agent.extract_answer_letter(raw)

        return {
            "pred_answer":      ans,
            "raw_output":       raw,
            "vote_history":     vote_history,
            "revision_history": revision_history,
            "syn_repo_history": syn_repo_history,
            "final_syn_report": syn_report,
        }

    # ── voting ───────────────────────────────────────────────
    def _vote(self, domain: str, syn_report: str) -> str:
        """Ask domain expert to vote YES or NO on the report."""
        system = f"You are a medical expert specialized in the {domain} domain."
        user   = (
            f"Here is a medical report:\n{syn_report}\n\n"
            f"As a medical expert specialized in {domain}, carefully read the report and "
            f"decide whether it is medically consistent, complete, and defensible.\n"
            f"Respond ONLY with one of the following exact words: YES, NO, or UNCLEAR."
        )
        raw = self.voter_agent.call(system=system, user=user)
        return self._parse_vote(raw)

    @staticmethod
    def _parse_vote(raw: str) -> str:
        if not raw or raw == "ERROR.":
            return "unclear"
        
        # Check first token for strict response
        first_word = raw.strip().split()[0].rstrip(".,:;!").upper()
        if first_word in ("YES", "NO", "UNCLEAR"):
            return first_word.lower()

        # Fallback substring search
        low = raw.lower()
        if re.search(r"\b(unclear|not sure|cannot tell|hard to tell|borderline|mixed|partially|partial)\b", low):
            return "unclear"
        if re.search(r"\b(no|not consistent|inconsistent|reject|disagree)\b", low):
            return "no"
        if re.search(r"\b(yes|consistent|agree|acceptable|good|looks good)\b", low):
            return "yes"

        return "unclear"

    # ── advice ───────────────────────────────────────────────
    def _get_advice(self, domain: str, syn_report: str) -> str:
        system = f"You are a medical expert specialized in the {domain} domain."
        user   = (
            f"Here is a medical report:\n{syn_report}\n\n"
            f"As a medical expert specialized in {domain}, evaluate whether the report is correct and complete.\n"
            f"If it is already adequate, briefly say so. If not, provide concise and specific revision advice that fixes the main issue.\n"
            f"Do not give vague comments; point to the exact problem and the correction needed.\n"
            f"Output EXACTLY in the format: '''Revisions: [proposed revision advice]'''"
        )
        return self.advice_agent.call(system=system, user=user)

    # ── revision ─────────────────────────────────────────────
    def _revise(
        self,
        syn_report: str,
        revision_advice: dict,
        question: str = "",
        options: str = "",
    ) -> str:
        user = f"Here is the original report:\n{syn_report}\n\n"
        for domain, advice in revision_advice.items():
            user += f"Advice from a medical expert specialized in {domain}: {advice}.\n"
        user += (
            f"\nBased on the above advice, revise the analysis carefully. "
            f"Preserve the correct parts, fix the main weakness, and output the revised analysis EXACTLY in "
            f"the format: '''Total Analysis: [revised analysis]'''"
        )
        raw = self.revision_agent.call(system="", user=user)
        # Extract revised Total Analysis text
        if "Total Analysis:" in raw:
            total = raw.split("Total Analysis:", 1)[1].strip()
        else:
            total = raw.strip()

        # Reconstruct structured report cleanly without losing question/options
        key_knowledge = ""
        if "Key Knowledge:" in syn_report and "Total Analysis:" in syn_report:
            part = syn_report.split("Total Analysis:", 1)[0]
            if "Key Knowledge:" in part:
                key_knowledge = part.split("Key Knowledge:", 1)[1].strip()

        report_parts = []
        if question:
            report_parts.append(f"Question: {question}")
        if options:
            report_parts.append(f"Options: {options}")
        if key_knowledge:
            report_parts.append(f"Key Knowledge: {key_knowledge}")
        report_parts.append(f"Total Analysis: {total}")

        return "\n".join(report_parts)

    # ── final answer ─────────────────────────────────────────
    def _derive_final_answer(self, syn_report: str, question: str = "", options: str = "") -> str:
        prompt_content = syn_report
        if options and "Options:" not in syn_report:
            prompt_content = f"Question: {question}\nOptions: {options}\n\n{syn_report}"

        user = (
            f"Here is a synthesised report:\n{prompt_content}\n\n"
            f"Based on the above report, select the optimal choice to answer the question.\n"
            f"Points to note:\n"
            f"1. The analyses provided should guide you towards the correct response.\n"
            f"2. Any option containing incorrect information cannot be the correct choice.\n"
            f"3. Please respond ONLY with the selected option's letter (A, B, C, D, or E) "
            f"using the format: '''Option: [Selected Option's Letter]'''"
        )
        return self.answer_agent.call(system="", user=user)

