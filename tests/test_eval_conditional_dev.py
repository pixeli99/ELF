import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from eval_conditional_dev import budget_probe_is_applicable


class ConditionalEvalBudgetProbeTests(unittest.TestCase):
    def test_probe_requires_a_clean_plan_endpoint(self):
        self.assertFalse(budget_probe_is_applicable(None, None))
        self.assertFalse(budget_probe_is_applicable("null", None))
        self.assertFalse(budget_probe_is_applicable("planning_first", 0.5))
        self.assertTrue(budget_probe_is_applicable("planning_first", 1.0))
        self.assertTrue(budget_probe_is_applicable("planning_first", 2.0))
        self.assertTrue(budget_probe_is_applicable("diagonal", None))
        self.assertTrue(budget_probe_is_applicable("endpoint_correct_lagging", None))


if __name__ == "__main__":
    unittest.main()
