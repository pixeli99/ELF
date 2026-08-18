#!/usr/bin/env python3
"""Single-GPU, single-process four-mode Ordered oracle content probe."""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from configs.config import load_config_from_yaml
from utils.stage_b_eval_runtime import attention_truth, build_clean_thinking_plan, load_model_and_encoder, load_thinking_plan_stack
from tools.eval_stage_b_common80k_generation import artifact_lock, checkpoint_gate, sampling_object, GPT2_SNAPSHOT
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from utils.metrics_utils import Metrics
from utils.sampling_utils import get_sampling_steps
from utils.stage_b_common80k_generation import sha256_file, tensor_sha256
from utils.stage_b_oracle_content_probe import read_pointer, select_probe_mapping
from utils.stage_b_oracle_serial import MODES, paired_bootstrap, validate_donor_mapping, validate_serial_protocol, validate_smoke_mapping

RUN = Path(os.environ.get("STAGE_B_ORDERED_RUN", ROOT / "outputs/elf_b_common80k_ordered_10k_v1"))
CKPT = RUN / "checkpoint_10000"
MAPPING = Path(os.environ.get("STAGE_B_ORACLE_MAPPING", ROOT / "data/stage_b_oracle/donor_mapping.jsonl"))
SHAPES = Path(os.environ.get("STAGE_B_SHAPES", ROOT / "data/stage_b_heldout_v2/stage_b_generation_shape_test_n1000_seed45_v2/shapes.jsonl"))


def log(stage, message):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] [serial_oracle] [{stage}] {message}", flush=True)


def _noise(seed, shape, scale):
    return torch.randn(shape, generator=torch.Generator().manual_seed(int(seed))) * scale


def _sampling(mode):
    if mode == "null":
        return sampling_object({"trajectory": "null", "alpha": 0.0})
    return sampling_object({"trajectory": "planning_first", "alpha": 2.0})


@torch.inference_mode()
def _rollout(model, cfg, mode, base, response_mask, plan_noise, plan, plan_mask,
             steps, sampling_seed, smoke):
    counter, sde_hashes, padding, plan_forward = {}, [], [], []
    selected = plan if mode.startswith("oracle_") else plan_noise
    override = None if not mode.startswith("oracle_") else [selected for _ in range(len(steps))]
    initial = plan_noise if not mode.startswith("oracle_") else selected
    hook_rows = []
    hook = None
    if smoke:
        def capture(_module, args, kwargs):
            xp = kwargs.get("x_plan")
            hook_rows.append({"response_sha256": tensor_sha256(args[0]),
                              "plan_sha256": None if xp is None else tensor_sha256(xp),
                              "plan_mask_sha256": tensor_sha256(kwargs["plan_mask"])})
        hook = model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        torch.manual_seed(int(sampling_seed)); torch.cuda.manual_seed_all(int(sampling_seed))
        result = _generate_samples_single_batch(
            model, torch.Generator().manual_seed(int(sampling_seed)), base.clone(), steps,
            None, None, cfg, _sampling(mode), 1.0, 3.0, record_plan=True,
            plan_override=override, plan_override_t=1.0 if override is not None else None,
            freeze_plan_override=override is not None, plan_mask=plan_mask,
            initial_plan_noise=initial, nfe_counter=counter,
            sde_noise_observer=lambda eps, t: sde_hashes.append(f"{float(t):.17g}:{tensor_sha256(eps)}"),
            response_attention_mask=response_mask, response_state_trace=padding,
            plan_forward_trace=plan_forward)
    finally:
        if hook is not None:
            hook.remove()
    latent, final_plan, trajectory = result
    if int(counter.get("model_forwards", 0)) != 32:
        raise AssertionError(f"{mode} actual NFE={counter}")
    if any(float(x) != 0.0 for x in padding):
        raise AssertionError(f"{mode} response padding became nonzero")
    if mode in ("null", "oracle_matched", "oracle_shuffled"):
        expected = tensor_sha256(selected)
        if any(tensor_sha256(x) != expected for x in trajectory):
            raise AssertionError(f"{mode} fixed plan changed")
    return latent, final_plan, {"sde_noise_hashes": sde_hashes, "padding_max_abs": padding,
                                "forward_inputs": hook_rows, "actual_nfe": 32}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, choices=(4, 1000), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    out = Path(a.output); work = Path(str(out) + ".work")
    if out.exists():
        raise FileExistsError(out)
    if work.exists() and not a.resume:
        raise FileExistsError(work)
    work.mkdir(parents=True, exist_ok=True)
    all_maps = [json.loads(x) for x in MAPPING.read_text().splitlines()]
    mapping_sha = validate_donor_mapping(all_maps)
    maps = select_probe_mapping(all_maps, a.samples)
    if a.samples == 4: validate_smoke_mapping(maps)
    shapes = {r["eval_id"]: r for r in map(json.loads, SHAPES.read_text().splitlines())}
    input_lock = {"mapping_sha256": mapping_sha, "mapping_file_sha256": sha256_file(MAPPING),
                  "shape_sha256": sha256_file(SHAPES), "checkpoint": checkpoint_gate(CKPT, "ordered"),
                  "mode_order": list(MODES), "samples": a.samples, "single_process": True}
    lock_path = work / "input_lock.json"
    if a.resume:
        if not lock_path.exists() or json.loads(lock_path.read_text()) != input_lock:
            raise ValueError("resume input lock mismatch")
    else:
        lock_path.write_text(json.dumps(input_lock, indent=2, sort_keys=True) + "\n")
    cfg = load_config_from_yaml(str(RUN / "config.yml")); device = torch.device("cuda")
    log("LOAD", "loading Ordered model and EMA exactly once")
    model, unused, tokenizer, _ = load_model_and_encoder(cfg, str(CKPT), device, load_encoder=False)
    if unused is not None: raise AssertionError("unexpected response encoder")
    log("LOAD", "loading frozen T5, 4-to-1 MLP and checkpoint whitener exactly once")
    t5, mlp = load_thinking_plan_stack(cfg, device)
    log("LOAD", "loading locked GPT-2 scorer exactly once")
    gpt_lock = artifact_lock(); metrics = Metrics(str(GPT2_SNAPSHOT), 2, 1024)
    validate_serial_protocol(MODES, {"model":1,"t5":1,"mlp":1,"whitener":1,"gpt2":1})
    progress = work / "progress.jsonl"; done = set(); records = []
    if a.resume and progress.exists():
        records = [json.loads(x) for x in progress.read_text().splitlines()]
        counts = {}
        for r in records: counts[r["eval_id"]] = counts.get(r["eval_id"], 0) + 1
        done = {k for k,v in counts.items() if v == 4}
        if any(v != 4 for v in counts.values()): raise ValueError("partial eval_id in resume progress")
    for index, meta in enumerate(maps):
        eid = meta["eval_id"]
        if eid in done:
            log("RESUME", f"skip completed eval_id={eid}"); continue
        shape = shapes[eid]; k = int(shape["K"]); length = int(shape["response_length"])
        matched, pm = build_clean_thinking_plan(meta, "recipient", tokenizer, t5, mlp, model, cfg, device)
        shuffled, donor_pm = build_clean_thinking_plan(meta, "donor", tokenizer, t5, mlp, model, cfg, device)
        if not torch.equal(pm, donor_pm) or tuple(matched.shape) != (1,k,512) or tensor_sha256(matched) == tensor_sha256(shuffled):
            raise AssertionError("matched/shuffled exact-K plan gate failed")
        width = int(cfg.max_length)
        base_cpu = torch.zeros((1,width,model.text_encoder_dim))
        base_cpu[:,:length] = _noise(shape["token_noise_seed"], (1,length,model.text_encoder_dim), cfg.denoiser_noise_scale)
        plan_noise = _noise(shape["plan_noise_seed"], (1,k,model.plan_latent_dim), cfg.denoiser_noise_scale).to(device)
        base = base_cpu.to(device); rm = torch.arange(width,device=device)[None,:] < length
        steps = get_sampling_steps(32,"logit_normal",cfg.denoiser_p_mean,cfg.denoiser_p_std,device=device,dtype=next(model.parameters()).dtype)
        attention = attention_truth(model,rm,pm)
        if not attention["response_reads_valid_plan"] or not attention["padding_keys_invisible"]:
            raise AssertionError("response-to-plan attention or padding visibility gate failed")
        base_hash = tensor_sha256(base); mode_rows=[]; latents={}; ids_by_mode={}; reference_sde=None
        for mode in MODES:
            selected = matched if mode == "oracle_matched" else shuffled if mode == "oracle_shuffled" else None
            latent, final_plan, trace = _rollout(model,cfg,mode,base,rm,plan_noise,selected,pm,steps,shape["sampling_seed"],a.samples==4)
            if reference_sde is None: reference_sde = trace["sde_noise_hashes"]
            elif trace["sde_noise_hashes"] != reference_sde: raise AssertionError("response SDE random stream differs across modes")
            if tensor_sha256(base) != base_hash: raise AssertionError("base response state mutated")
            tpd = 0.0 if mode == "null" else 1.0
            ids = _dlm_decode_batch(latent,model,steps[-1].item(),cfg,3.0,x_plan=final_plan,t_plan_decode_val=tpd,
                                    plan_trajectory=_sampling(mode).plan_trajectory,plan_mask=pm,attention_mask=rm)
            eos=tokenizer.eos_token_id or 1; pad=tokenizer.pad_token_id or 0; raw=ids.clone()
            ids=mask_after_eos(ids,eos,pad);ids=torch.where(rm,ids,torch.full_like(ids,pad))
            text=tokenizer.decode(ids[0].cpu().numpy(),skip_special_tokens=True)
            latents[mode]=latent.detach().cpu();ids_by_mode[mode]=ids.detach().cpu()
            mode_rows.append({"eval_id":eid,"mode":mode,"K":k,"response_length":length,"nonempty":bool(text.strip()),
                              "eos":bool(((raw==eos)&rm).any()),"generated_length":int((ids!=pad).sum()),"text":text,
                              "response_initial_noise_sha256":base_hash,"response_mask_sha256":tensor_sha256(rm),
                              "plan_input_sha256":tensor_sha256(plan_noise if selected is None else selected),
                              "sde_schedule_sha256":hashlib.sha256(json.dumps(reference_sde).encode()).hexdigest(),
                              "t_grid_sha256":tensor_sha256(steps),"actual_nfe":trace["actual_nfe"],
                              "padding_max_abs":max(trace["padding_max_abs"] or [0.0]),
                              "response_reads_valid_plan":attention["response_reads_valid_plan"],
                              "padding_keys_invisible":attention["padding_keys_invisible"]})
        valid = rm.detach().cpu().unsqueeze(-1)
        rel = float((((latents["oracle_matched"]-latents["oracle_shuffled"]).square()*valid).sum()/
                     ((latents["oracle_matched"].square()*valid).sum().clamp_min(1e-12))).sqrt())
        change = float((((ids_by_mode["oracle_matched"]!=ids_by_mode["oracle_shuffled"]) & rm.cpu()).sum()/rm.sum().cpu()).item())
        for r in mode_rows:
            r["matched_shuffled_final_response_relative_l2"] = rel
            r["matched_shuffled_token_change_fraction"] = change
            with progress.open("a") as f: f.write(json.dumps(r,sort_keys=True)+"\n")
        records.extend(mode_rows); log("GENERATE",f"{index+1}/{a.samples} eval_id={eid} four modes complete")
    # Score one loaded GPT-2 instance; preserve every record, including empty outputs.
    scoreable=[i for i,r in enumerate(records) if r["nonempty"]]
    score=metrics.record_generative_perplexity([records[i]["text"] for i in scoreable],1024,True)
    for pos,i in enumerate(scoreable):
        n=int(score["per_sample_token_count"][pos]);v=float(score["per_sample_nll_sum"][pos])
        records[i].update(gpt2_token_count=n,gpt2_nll_sum=v,token_normalized_nll=v/n,
                          token_frequency_entropy=float(score["per_sample_token_frequency_entropy"][pos]))
    for r in records: r.pop("text",None)
    with (work/"per_sample.jsonl").open("w") as f:
        for r in records:f.write(json.dumps(r,sort_keys=True)+"\n")
    summaries={}
    for mode in MODES:
        rr=[r for r in records if r["mode"]==mode and r.get("gpt2_token_count",0)>0]
        total_nll=sum(r["gpt2_nll_sum"] for r in rr);total_tok=sum(r["gpt2_token_count"] for r in rr)
        summaries[mode]={"samples":a.samples,"valid_samples":len(rr),"corpus_gen_ppl":math.exp(total_nll/total_tok),
                         "mean_token_frequency_entropy":sum(r["token_frequency_entropy"] for r in rr)/len(rr),
                         "eos_rate":sum(r["eos"] for r in records if r["mode"]==mode)/a.samples,
                         "mean_length":sum(r["generated_length"] for r in records if r["mode"]==mode)/a.samples}
    matched={r["eval_id"]:r for r in records if r["mode"]=="oracle_matched" and "token_normalized_nll" in r}
    shuffled={r["eval_id"]:r for r in records if r["mode"]=="oracle_shuffled" and "token_normalized_nll" in r}
    common=sorted(set(matched)&set(shuffled))
    paired=paired_bootstrap([matched[x]["token_normalized_nll"] for x in common],[shuffled[x]["token_normalized_nll"] for x in common])
    summary={"complete":True,"single_gpu":True,"single_process":True,"mode_order":list(MODES),"samples":a.samples,
             "metrics":summaries,"matched_minus_shuffled":paired,"mapping_sha256":mapping_sha,
             "checkpoint_sha256":input_lock["checkpoint"]["sha256"],"gpt2_lock":gpt_lock,
             "old_consistency_computed":False,"extra_matched_reference":False,"cross_gpu_gate":False}
    (work/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (work/"manifest.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (work/"exit_code").write_text("0\n")
    (work/"sha256sums.txt").write_text("".join(f"{sha256_file(x)}  {x.name}\n" for x in sorted(work.iterdir()) if x.name!="sha256sums.txt"))
    os.rename(work,out);log("PASS",f"ORACLE_SERIAL_GATE_PASS output={out}")

if __name__ == "__main__": main()
