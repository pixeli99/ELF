# Oracle Plan Eval

Smoke test:

```bash
python tools/eval_oracle_plan.py \
  --config_path src/configs/training_configs/train_owt_ELF-B_ordered.yml \
  --checkpoint_path outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592 \
  --sampling_config_path src/configs/sampling_configs/ordered_sampling_configs.yml \
  --dataset embedded-language-flows/openwebtext-t5 \
  --num_samples 4 \
  --num_sampling_steps 4 \
  --modes null,self_planning_first,oracle_matched,oracle_shuffled \
  --seed 42 \
  --global_batch_size 2 \
  --eval_ppl_batch_size 2 \
  --out_dir results/oracle_plan_eval/smoke_n4_steps4_seed42
```

Null parity smoke (matches the official rank-0 batch size and 32-step sampling setup):

```bash
CUDA_VISIBLE_DEVICES=4 python tools/eval_oracle_plan.py \
  --config_path src/configs/training_configs/train_owt_ELF-B_ordered.yml \
  --checkpoint_path outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592 \
  --sampling_config_path src/configs/sampling_configs/ordered_sampling_configs.yml \
  --dataset embedded-language-flows/openwebtext-t5 \
  --num_samples 64 \
  --num_sampling_steps 32 \
  --modes null \
  --seed 42 \
  --global_batch_size 8 \
  --eval_ppl_batch_size 4 \
  --out_dir results/oracle_plan_eval/parity_null_n64_steps32_seed42
```

Full run:

```bash
python tools/eval_oracle_plan.py \
  --config_path src/configs/training_configs/train_owt_ELF-B_ordered.yml \
  --checkpoint_path outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592 \
  --sampling_config_path src/configs/sampling_configs/ordered_sampling_configs.yml \
  --dataset embedded-language-flows/openwebtext-t5 \
  --num_samples 256 \
  --num_sampling_steps 32 \
  --modes null,self_planning_first,oracle_matched,oracle_shuffled \
  --seed 42 \
  --global_batch_size 4 \
  --eval_ppl_batch_size 4 \
  --out_dir results/oracle_plan_eval/n256_steps32_seed42
```
