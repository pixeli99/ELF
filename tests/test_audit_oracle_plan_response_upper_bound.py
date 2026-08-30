import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_oracle_plan_response_upper_bound import resolve_schedule_window


def test_schedule_window_extends_the_deterministic_training_prefix():
    result = resolve_schedule_window(
        dataset_rows=474460,
        configured_schedule_rows=80000,
        checkpoint_step=80000,
        global_batch_size=1,
        start_index=80000,
        candidate_count=1024,
    )
    assert result == {
        "checkpoint_seen_presentation_rows": 80000,
        "candidate_stop": 81024,
        "audit_schedule_rows": 81024,
    }


def test_schedule_window_rejects_checkpoint_overlap():
    with pytest.raises(ValueError, match="overlaps rows already seen"):
        resolve_schedule_window(
            dataset_rows=474460,
            configured_schedule_rows=80000,
            checkpoint_step=80000,
            global_batch_size=1,
            start_index=79999,
            candidate_count=512,
        )


def test_schedule_window_preserves_partial_checkpoint_audit():
    result = resolve_schedule_window(
        dataset_rows=474460,
        configured_schedule_rows=80000,
        checkpoint_step=40000,
        global_batch_size=1,
        start_index=40000,
        candidate_count=1024,
    )
    assert result["audit_schedule_rows"] == 80000
    assert result["checkpoint_seen_presentation_rows"] == 40000
