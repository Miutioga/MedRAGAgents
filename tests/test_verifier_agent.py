import unittest

from agents.verifier_agent import VerifierAgent


class VerifierAgentTests(unittest.TestCase):
    def test_parse_vote_handles_yes_no_and_unclear(self):
        self.assertEqual(VerifierAgent._parse_vote("YES"), "yes")
        self.assertEqual(VerifierAgent._parse_vote("NO"), "no")
        self.assertEqual(VerifierAgent._parse_vote("UNCLEAR"), "unclear")
        self.assertEqual(VerifierAgent._parse_vote("I am not sure"), "unclear")

    def test_parse_vote_prefers_explicit_yes_when_both_present(self):
        self.assertEqual(VerifierAgent._parse_vote("YES, but I have concerns"), "yes")
        self.assertEqual(VerifierAgent._parse_vote("NO, but maybe"), "no")


if __name__ == "__main__":
    unittest.main()
