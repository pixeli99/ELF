import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_conditional_evals import paired_bootstrap


class PairedConditionalEvalTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_uses_left_minus_right(self):
        values = [1.0, 2.0, 3.0, 4.0]
        first = paired_bootstrap(values, seed=7, draws=1000)
        second = paired_bootstrap(values, seed=7, draws=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 2.5)
        self.assertEqual(first["paired_sample_win_rate"], 1.0)
        self.assertEqual(first["bootstrap_probability_gt_zero"], 1.0)

    def test_bootstrap_rejects_nonfinite_or_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite nonempty"):
            paired_bootstrap([])
        with self.assertRaisesRegex(ValueError, "finite nonempty"):
            paired_bootstrap([float("nan")])


if __name__ == "__main__":
    unittest.main()
