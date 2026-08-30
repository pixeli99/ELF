import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_training_log_intervals import compare_logs


def _line(step, loss, response_tokens=100, plan_slots=640, mediated=None):
    record = {
        "step": str(step), "loss": str(loss), "l2": str(loss - 0.2),
        "ce": "0.1", "plan": "0.1", "grad": "0.2",
        "response_tokens": str(response_tokens), "plan_slots": str(plan_slots),
    }
    if mediated is not None:
        record["mediated_rows"] = str(mediated)
    return f"progress INFO - __main__ - {record}\n"


def test_paired_log_comparison(tmp_path):
    baseline = tmp_path / "baseline.log"
    branch = tmp_path / "branch.log"
    baseline.write_text(_line(10, 0.5) + _line(20, 0.6))
    branch.write_text(_line(10, 0.7, mediated=4) + _line(20, 0.9, mediated=6))
    payload = compare_logs(baseline, branch, 10, 20, 10)
    assert payload["paired_intervals"] == 2
    assert payload["mediation_rate"] == 0.5
    assert payload["overall"]["mean_branch_minus_baseline"]["total_loss"] == pytest.approx(0.25)
    assert payload["overall"]["mean_branch_minus_baseline"]["plan_loss"] == 0.0


def test_paired_log_comparison_rejects_mismatched_schedule(tmp_path):
    baseline = tmp_path / "baseline.log"
    branch = tmp_path / "branch.log"
    baseline.write_text(_line(10, 0.5, response_tokens=100))
    branch.write_text(_line(10, 0.6, response_tokens=101, mediated=5))
    with pytest.raises(ValueError, match="response-token mismatch"):
        compare_logs(baseline, branch, 10, 10, 10)
