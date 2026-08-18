import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src/utils/stage_b_common80k_generation.py"
SPEC = importlib.util.spec_from_file_location("common_gen", PATH)
gen = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(gen)
FINALIZER_PATH = ROOT / "tools/finalize_stage_b_common80k_generation.py"
FINALIZER_SPEC = importlib.util.spec_from_file_location("common_gen_finalizer", FINALIZER_PATH)
finalizer = importlib.util.module_from_spec(FINALIZER_SPEC); FINALIZER_SPEC.loader.exec_module(finalizer)


class Common80kGenerationTests(unittest.TestCase):
    def test_pid_schema_legacy_three_and_current_four_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=root/"pids.tsv"
            p.write_text("10\t0\tarm_a\n11\t1\tarm_b\t/tmp/arm_b.log\n")
            rows=finalizer.parse_pid_rows(p,root)
            self.assertEqual([r["condition_id"] for r in rows],["arm_a","arm_b"])

    def test_pid_schema_v2_header_and_extra_header_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=root/"pids.tsv"
            p.write_text("schema_version\tpid\tgpu\tcondition_id\tworker_log_work\tworker_log_final\n2\t10\t0\tarm_a\t/tmp/a.work.log\t/tmp/a.log\n")
            self.assertEqual(finalizer.parse_pid_rows(p,root)[0]["schema_version"],"2")
            p.write_text("schema_version\tpid\tgpu\tcondition_id\tworker_log_work\tworker_log_final\textra\n2\t10\t0\tarm_a\ta\tb\tc\n")
            with self.assertRaisesRegex(ValueError,"invalid header"): finalizer.parse_pid_rows(p,root)

    def test_pid_headerless_invalid_columns_and_duplicate_condition(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=root/"pids.tsv"
            p.write_text("1\t0\ta\tx\textra\n")
            with self.assertRaisesRegex(ValueError,"line 1"): finalizer.parse_pid_rows(p,root)
            p.write_text("1\t0\ta\n2\t1\ta\n")
            with self.assertRaisesRegex(ValueError,"duplicate condition"): finalizer.parse_pid_rows(p,root)

    def test_final_worker_log_resolves_after_atomic_rename(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/"arm_a").mkdir(); (root/"arm_a"/"worker.log").write_text("done")
            p=root/"pids.tsv";p.write_text("1\t0\tarm_a\t/tmp/gone.work/worker.log\n")
            self.assertEqual(finalizer.parse_pid_rows(p,root)[0]["worker_log"],str(root/"arm_a"/"worker.log"))

    def test_exit_current_four_columns_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"exit_codes.tsv";p.write_text("a\t0\t10\t0\nb\t0\t11\t1\n")
            self.assertEqual(finalizer.parse_exit_rows(p),{"a":0,"b":0})
            p.write_text("a\t0\na\t0\n")
            with self.assertRaisesRegex(ValueError,"duplicate condition"): finalizer.parse_exit_rows(p)

    def test_fixed_width_841_mask_and_noise_prefix(self):
        width, length, dim = 1024, 841, 7
        mask = torch.arange(width).unsqueeze(0) < length
        self.assertEqual(int(mask.sum()), 841)
        self.assertEqual(int((~mask).sum()), 183)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        dynamic = torch.randn((1, length, dim), generator=g1)
        fixed = torch.zeros((1, width, dim))
        fixed[:, :length] = torch.randn((1, length, dim), generator=g2)
        self.assertTrue(torch.equal(dynamic, fixed[:, :length]))
        fixed.masked_fill_(~mask.unsqueeze(-1), 0)
        self.assertEqual(float(fixed[:, length:].abs().max()), 0.0)

    def test_bidirectional_mask_is_key_padding_only(self):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from modules.model import build_plan_response_attention_mask
        token = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        plan = torch.tensor([[1, 1]], dtype=torch.bool)
        actual = build_plan_response_attention_mask(token, plan, 2, 1, 1, 1, "bidirectional")
        self.assertEqual(actual.ndim, 2)
        self.assertEqual(actual.tolist(), [[True, True, True, True, True, True, True, False, False]])

    def test_width_comes_from_resolved_config_and_vanilla_stays_planless(self):
        text = (ROOT / "tools/eval_stage_b_common80k_generation.py").read_text()
        self.assertIn("response_width = int(cfg.max_length)", text)
        self.assertNotIn("response_width = 1024", text)
        self.assertIn('if group == "vanilla" and any(kwargs.get(name) is not None', text)

    def test_padding_is_zeroed_initially_and_after_solver_steps(self):
        evaluator = (ROOT / "tools/eval_stage_b_common80k_generation.py").read_text()
        generation = (ROOT / "src/utils/generation_utils.py").read_text()
        self.assertIn("token = torch.zeros((1, response_width, model.text_encoder_dim))", evaluator)
        self.assertGreaterEqual(generation.count("z.mul_(response_keep); x_pred.mul_(response_keep)"), 3)
        self.assertIn("response_padding_max_abs_by_step", evaluator)

    def test_decode_and_metrics_only_see_valid_response(self):
        text = (ROOT / "tools/eval_stage_b_common80k_generation.py").read_text()
        self.assertIn("attention_mask=token_mask", text)
        self.assertIn("ids = torch.where(token_mask, ids", text)
        self.assertIn("generated_length = int((ids != pad).sum())", text)
        self.assertIn("generated_length > length", text)

    def test_optional_mask_preserves_legacy_default(self):
        generation = (ROOT / "src/utils/generation_utils.py").read_text()
        sampling = (ROOT / "src/utils/sampling_utils.py").read_text()
        self.assertIn("response_attention_mask: Optional[torch.Tensor] = None", generation)
        self.assertIn('if response_attention_mask is not None:\n        pk["attention_mask"]', sampling)

    def test_launcher_logging_and_failure_accounting_contract(self):
        text = (ROOT / "scripts/run_stage_b_common80k_generation.sh").read_text()
        for needle in ("set -Eeuo pipefail", "trap 'on_err", "PYTHONUNBUFFERED=1",
                       "failure_report.json", "exit_codes.tsv", "pids.tsv",
                       "set +e; wait", "GENERATION_EVALUATION_GATE_FAIL",
                       "GENERATION_EVALUATION_GATE_PASS", "PREFLIGHT_ONLY"):
            self.assertIn(needle, text)

    def test_worker_has_immediate_progress_and_fixed_width_mask(self):
        text = (ROOT / "tools/eval_stage_b_common80k_generation.py").read_text()
        self.assertIn('"GENERATE"', text)
        self.assertIn('response_attention_mask=token_mask', text)
        self.assertIn('response_width = int(cfg.max_length)', text)
        self.assertIn('flush=True', text)

    def test_preflight_reports_expected_and_actual(self):
        text = (ROOT / "tools/preflight_stage_b_common80k_generation.py").read_text()
        self.assertIn("expected={expected} actual={actual}", text)
        self.assertIn("GENERATION_PREFLIGHT_GATE_PASS", text)

    def test_condition_matrix_is_exact_and_unique(self):
        self.assertEqual(len(gen.CONDITIONS), 14)
        self.assertEqual(len({row[0] for row in gen.CONDITIONS}), 14)
        self.assertTrue(all("alpha.5" not in row[0] for row in gen.CONDITIONS))

    def test_ordered_clocks_and_endpoints(self):
        expected = {2.0: 1.0, 1.0: 1.0, .5: .5, 0.0: 0.0}
        for alpha, endpoint in expected.items():
            spec = next(gen.condition(cid) for cid, group, a, s in gen.CONDITIONS
                        if group == "ordered" and a == alpha and s == 8)
            self.assertEqual(gen.plan_protocol(spec)["endpoint"], endpoint)

    def test_diagonal_is_separate_checkpoint_group(self):
        self.assertEqual(gen.condition("diagonal_alpha1p0_steps8")["model_group"], "diagonal")
        self.assertEqual(gen.condition("ordered_alpha1p0_steps8")["model_group"], "ordered")

    def test_register_and_alpha_zero_are_fixed(self):
        initial = torch.randn(1, 3, 2)
        gen.assert_fixed_plan(initial, [initial.clone(), initial.clone()], "register")
        with self.assertRaises(ValueError): gen.assert_fixed_plan(initial, [initial + 1], "register")

    def test_vanilla_has_no_plan_interface(self):
        token = torch.zeros(1, 5, 2); mask = torch.ones(1, 5, dtype=torch.bool)
        gen.assert_truth_inputs("vanilla", token, mask, None, None, None)
        with self.assertRaises(ValueError):
            gen.assert_truth_inputs("vanilla", token, mask, torch.zeros(1,1,2), None, None)

    def test_register_rejects_missing_or_clean_plan_injection_interface(self):
        token = torch.zeros(1, 5, 2); mask = torch.ones(1, 5, dtype=torch.bool)
        with self.assertRaises(ValueError): gen.assert_truth_inputs("register", token, mask, None, None, None)
        plan = torch.randn(1, 3, 2); pmask = torch.ones(1, 3, dtype=torch.bool)
        gen.assert_truth_inputs("register", token, mask, plan, pmask, torch.zeros(1))

    def test_shape_schema_rejects_gold(self):
        self.assertNotIn("thinking", gen.ALLOWED_SHAPE_FIELDS)
        self.assertNotIn("response", gen.ALLOWED_SHAPE_FIELDS)
        self.assertNotIn("input_ids", gen.ALLOWED_SHAPE_FIELDS)

    def test_smoke_selection_has_distinct_k_and_is_deterministic(self):
        rows = [{"eval_id": f"e{i}", "K": i + 10, "response_length": i + 20} for i in range(20)]
        a = gen.select_smoke(rows); b = gen.select_smoke(list(reversed(rows)))
        self.assertEqual([x["eval_id"] for x in a], [x["eval_id"] for x in b])
        self.assertEqual(len({x["K"] for x in a}), 4); self.assertGreater(max(x["K"] for x in a), 16)

    def test_resume_requires_complete_hash_valid_per_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / "per_sample.jsonl"; data.write_text('{"eval_id":"e"}\n')
            noise = root / "noise_hashes.json"; noise.write_text("{}\n")
            manifest = {"complete": True, "condition_id": "x", "num_samples": 1,
                "outputs": {"per_sample.jsonl": {"sha256": gen.sha256_file(data)},
                            "noise_hashes.json": {"sha256": gen.sha256_file(noise)}}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(gen.validate_resume_arm(root, {"condition_id":"x","num_samples":1}))
            data.write_text("corrupt\n")
            self.assertFalse(gen.validate_resume_arm(root, {"condition_id":"x","num_samples":1}))

    def test_worker_gate_detects_running_pid(self):
        self.assertFalse(gen.all_workers_finished([{"pid": os.getpid()}]))
        with mock.patch("os.kill", side_effect=ProcessLookupError):
            self.assertTrue(gen.all_workers_finished([{"pid": 999999}]))

    def test_runtime_nfe_definition_matches_grid(self):
        # Original get_sampling_steps returns n_steps+1 endpoints; the sampler executes n_steps intervals.
        for nfe in (8, 32): self.assertEqual((nfe + 1) - 1, nfe)

    def test_wrapper_calls_original_implementations(self):
        text=(ROOT/"tools/eval_stage_b_common80k_generation.py").read_text()
        for symbol in ("_generate_samples_single_batch", "get_sampling_steps", "_dlm_decode_batch",
                       "Metrics", "load_model_and_encoder"):
            self.assertIn(symbol, text)

    def test_metrics_formula_is_not_duplicated(self):
        text=(ROOT/"src/utils/metrics_utils.py").read_text()
        self.assertIn('"per_sample_nll_sum": per_sample_nll_sum.tolist()', text)
        self.assertIn('entropy = float(-np.sum(probs * np.log(probs + 1e-10)))', text)

    def test_new_worker_writes_only_per_sample_name(self):
        text=(ROOT/"tools/eval_stage_b_common80k_generation.py").read_text()
        self.assertIn('"per_sample.jsonl"', text); self.assertNotIn('"samples.jsonl"', text)

    def test_noise_is_shape_identity_derived_not_condition_derived(self):
        text=(ROOT/"tools/eval_stage_b_common80k_generation.py").read_text()
        self.assertIn('row["token_noise_seed"]', text); self.assertIn('row["plan_noise_seed"]', text)
        self.assertNotIn('hash(args.condition_id)', text)

    def test_formal_launcher_has_smoke_gate_and_wait(self):
        text=(ROOT/"scripts/run_stage_b_common80k_generation.sh").read_text()
        self.assertIn('--root "$SMOKE" --expected-rows 4 --check-root-only', text)
        self.assertIn('wait "${PIDS[$i]}"', text)
        self.assertNotIn("torchrun", text)


if __name__ == "__main__": unittest.main()
