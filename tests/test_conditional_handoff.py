import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from configs.config import load_config_from_yaml
from utils.data_utils import get_dataloader, load_jsonl_dataset
from utils.sampling_utils import restore_cond
from utils.stage_b_common80k_generation import assert_fixed_plan


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 97 + 1 for char in text]}


class ConditionalHandoffTests(unittest.TestCase):
    def test_upstream_conditional_configs_parse(self):
        for name in ("train_de-en_ELF-B.yml", "train_xsum_ELF-B.yml"):
            config = load_config_from_yaml(str(ROOT / "src/configs/training_configs" / name))
            self.assertEqual(config.sampling_configs_path,
                             "src/configs/sampling_configs/cond_sampling_configs.yml")
            self.assertTrue(config.sampling_configs)

    def test_condition_mask_positions_and_clean_prefix_visibility(self):
        dataset = [{"condition_input_ids": [11, 12], "input_ids": [21, 22, 23]}]
        batch = next(iter(get_dataloader(
            dataset, batch_size=1, shuffle=False, drop_last=False,
            max_seq_length=6, max_input_seq_length=4, distributed=False,
        )))
        np.testing.assert_array_equal(batch["cond_seq_mask"], [[1, 1, 0, 0, 0, 0]])
        np.testing.assert_array_equal(batch["attention_mask"], [[1, 1, 1, 1, 1, 0]])
        encoder_mask = batch["encoder_attention_mask"][0]
        self.assertEqual(encoder_mask[0].tolist(), [1, 1, 0, 0, 0, 0])
        self.assertEqual(encoder_mask[2].tolist(), [1, 1, 1, 1, 1, 0])

    def test_all_groups_share_condition_and_keep_plan_semantics(self):
        condition = torch.randn(2, 7, 8)
        mask = torch.zeros(2, 7, 1)
        mask[:, :3] = 1
        noisy = torch.randn_like(condition)
        restored = {group: restore_cond(noisy.clone(), condition, mask)
                    for group in ("ordered", "diagonal", "register", "vanilla")}
        for value in restored.values():
            torch.testing.assert_close(value[:, :3], condition[:, :3])
        configs = {
            group: load_config_from_yaml(str(
                ROOT / "src/configs/training_configs" /
                f"train_stage_b_common80k_{group}_10k_v1.yml"))
            for group in ("ordered", "diagonal", "register", "vanilla")
        }
        self.assertEqual(configs["vanilla"].num_plan_slots, 0)
        self.assertTrue(configs["register"].plan_register_only)
        self.assertEqual(configs["ordered"].group_mode, "ordered")
        self.assertEqual(configs["diagonal"].group_mode, "diagonal")
        self.assertGreater(configs["ordered"].max_plan_slots, 16)
        self.assertEqual(configs["ordered"].max_plan_slots, configs["diagonal"].max_plan_slots)

    def test_register_plan_is_frozen(self):
        plan = torch.randn(1, 9, 4)
        assert_fixed_plan(plan, [plan.clone() for _ in range(8)], "register")
        with self.assertRaises(ValueError):
            assert_fixed_plan(plan, [plan.clone(), plan + 1], "register")

    def test_eval_input_has_instruction_but_no_gold_thinking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            path.write_text(json.dumps({"input": "instruction", "output": "reference"}) + "\n")
            rows = load_jsonl_dataset(path, _Tokenizer())
        self.assertEqual(set(rows[0]), {"index", "input", "target", "condition_input_ids", "input_ids"})
        self.assertNotIn("thinking", rows[0])
        self.assertNotIn("plan_input_ids", rows[0])

    def test_handoff_runtime_is_current_thinking_plan_not_legacy_probe(self):
        runtime = (ROOT / "src/utils/stage_b_eval_runtime.py").read_text()
        helper = (ROOT / "src/utils/stage_b_oracle_content_probe.py").read_text()
        self.assertIn('checkpoint["ema_params1"]', runtime)
        self.assertIn("build_thinking_plan_target", runtime)
        self.assertIn("ThinkingMLPEncoder", runtime)
        self.assertIn("apply_plan_whitening", runtime)
        self.assertNotIn("build_plan_target", runtime)
        self.assertNotIn("plan_probes", runtime + helper)
        self.assertNotIn("consistency", helper)


if __name__ == "__main__":
    unittest.main()
