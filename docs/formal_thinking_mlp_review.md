# Formal Thinking MLP 4-to-1 Autoencoder

This module trains a standalone compression autoencoder for thinking trajectories. It is
trained before ELF and its encoder is frozen for later use by an ELF plan stream.

## Design

Frozen `t5-small` produces normalized contextual thinking latents `[B, L, 512]`. The
formal boundary converts BF16 T5 output to FP32. Adjacent groups of four token vectors
are zero-padded and concatenated:

```text
[B, L, 512] -> [B, K, 4, 512] -> [B, K, 2048]
K_i = ceil(L_i / 4)
```

`reconstruction_mask[B,K,4]` marks real token vectors and `plan_mask[B,K]` marks groups
with at least one real token. Tail padding never contributes to training or statistics.

The encoder is `Linear(2048,6144) -> GELU -> Linear(6144,512)` and has 15,735,296
parameters. The audit-only decoder is `Linear(512,6144) -> GELU -> Linear(6144,2048)`
and has 15,736,832 parameters. Total trainable parameters are 31,472,128.

The only training objective is FP32 masked reconstruction MSE: feature MSE is averaged
within each valid token vector, then averaged over valid thinking tokens. The decoder has
no skip connection and cannot read the source latent or response.

## Offline Lifecycle

1. Train the autoencoder offline with frozen, deterministic T5 encoding.
2. Select `best.pt` only by validation reconstruction MSE.
3. Export a weights-only `frozen_encoder.pt`; Stage-B uses this encoder and never uses
   the decoder.
4. Compute channel-wise plan-slot mean and standard deviation over valid train-split
   slots with streaming FP64 moments, then freeze that whitening transform.

Tokenization uses T5 special-token semantics, an explicit maximum length of 1024, and
no truncation. Complete training checkpoints are trusted local pickle artifacts loaded
with `weights_only=False`; the exported encoder contains only tensors/basic fields and is
loaded with `weights_only=True`.

## Engineering Smoke Results

The smoke dataset is only an engineering validation and is not a scientific training
corpus.

| Metric | Value |
|---|---:|
| Validation reconstruction MSE, MLP | 0.139537 |
| Validation reconstruction MSE, mean-pool | 0.352550 |
| Clean T5 decoder CE / accuracy | 0.000367 / 99.9935% |
| MLP reconstruction decoder CE / accuracy | 0.002965 / 99.9391% |
| Mean-pool decoder CE / accuracy | 2.195745 / 62.2443% |

These diagnostics show reconstruction and frozen-decoder token recoverability. They do
not establish reasoning semantics, sample-specific planning value, or final generation
quality.
