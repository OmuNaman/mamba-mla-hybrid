# Results Summary — mamba-mla-hybrid

## Track-A pretrain perplexity (FineWeb-Edu val / WikiText-103 val)

| size | variant | ppl_fineweb (mean±std, n) | ppl_wikitext |
|---|---|---|---|
| A1:r3 | mamba_mla | 74.38 (n=1) | 308.11 (n=1) |
| A1:r7 | mamba_mla | 72.05 (n=1) | 300.18 (n=1) |
| M | mamba_full | 126.48 (n=1) | 686.45 (n=1) |
| M | mamba_mla | 120.00 (n=1) | 662.80 (n=1) |
| M | mamba_swa | 125.78 (n=1) | 796.59 (n=1) |
| S | mamba2 | 74.54 (n=1) | 316.66 (n=1) |
| S | mamba_full | 70.34±1.96 (n=3) | 288.84±12.53 (n=3) |
| S | mamba_mla | 71.00±1.75 (n=3) | 299.12±12.93 (n=3) |
| S | mamba_swa | 69.79±2.82 (n=3) | 285.66±23.11 (n=3) |
| S | transformer | 96.19 (n=1) | 454.38 (n=1) |
| S | transformer_mla | 106.77 (n=1) | 509.98 (n=1) |

### A1 ratio ablation (mamba_mla, size-S)

| run | attn_ratio | ppl_fineweb |
|---|---|---|
| A1_mla_r3 | 3 | 74.38 |
| A1_mla_r7 | 7 | 72.05 |

## Track-B MQAR baselines — dense recall accuracy by variant (mean over n_kv, seeds)

| variant | MQAR acc (mean±std, n) |
|---|---|
| transformer | 0.85±0.26 (n=12) |
| mamba2 | 0.96±0.10 (n=12) |
| mamba_swa | 0.75±0.43 (n=12) |
| mamba_full | 0.78±0.39 (n=12) |
| mamba_mla | 0.74±0.42 (n=12) |

## Track-B H4 — MQAR accuracy vs MLA latent dim d_c (mamba_mla; mean over n_kv, seeds)

| d_c | MQAR acc (mean±std, n) |
|---|---|
| 16 | 0.83±0.37 (n=12) |
| 32 | 0.67±0.47 (n=12) |
| 64 | 0.85±0.33 (n=12) |
| 128 | 0.76±0.41 (n=12) |
| 256 | 0.67±0.45 (n=12) |
| 512 | 0.93±0.23 (n=12) |

## Efficiency — KV/latent cache bytes per sequence vs context length

| variant(size) | cache@4k (MB) | @16k | @64k | mamba_state(MB) |
|---|---|---|---|---|
| S_transformer_s0 | 151 | 604 | 2416 | 0 |
| S_mamba2_s0 | 5 | 5 | 5 | 5 |
| S_mamba_full_s0 | 29 | 105 | 407 | 4 |
| S_mamba_swa_s0 | 7 | 7 | 7 | 4 |
| S_mamba_mla_s0 | 9 | 23 | 80 | 4 |
| S_transformer_mla_s0 | 28 | 113 | 453 | 0 |
| M_mamba_full_s0 | 78 | 279 | 1085 | 11 |
| M_mamba_swa_s0 | 19 | 19 | 19 | 11 |
| M_mamba_mla_s0 | 20 | 49 | 162 | 11 |
| A1_mla_r3 | 11 | 32 | 117 | 4 |
| A1_mla_r7 | 9 | 23 | 80 | 4 |

## recall_eval — passkey accuracy vs length (checkpoints pretrained @ctx 2048, light-adapted @len 256, evaluated 512–4096 ≈ up to 2× pretrain-ctx / 16× adapt-len; extrapolation)

> Note (n per cell): mamba_full/swa/mla have 3 seeds (s0/s1/s2 rows); transformer, mamba2, transformer_mla
> are n=1. The headline contrast (mamba_mla holds vs pure transformer 0.06 @4096) pairs a 3-seed arm
> against a single transformer seed — read it as a qualitative "hybrid extrapolates, dense MHA degrades"
> observation, not as a precise gap (the transformer 0.06 is likely RoPE length-extrapolation, not a
> recall-capability deficit). See `critique/experiment_review.md` claims-contract.

| variant | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| transformer_s0 | 0.98 | 0.95 | 0.62 | 0.06 |
| mamba2_s0 | 0.84 | 0.79 | 0.72 | 0.57 |
| mamba_full_s0 | 1.00 | 1.00 | 0.98 | 0.76 |
| mamba_swa_s0 | 1.00 | 0.84 | 0.71 | 0.64 |
| mamba_mla_s0 | 1.00 | 1.00 | 0.97 | 0.98 |
| transformer_mla_s0 | 1.00 | 0.98 | 0.96 | 0.87 |
| mamba_full_s1 | 0.96 | 0.96 | 0.91 | 0.86 |
| mamba_full_s2 | 1.00 | 1.00 | 1.00 | 0.92 |
| mamba_swa_s1 | 0.99 | 0.77 | 0.64 | 0.60 |
| mamba_swa_s2 | 1.00 | 0.88 | 0.75 | 0.76 |
| mamba_mla_s1 | 1.00 | 0.99 | 0.99 | 0.98 |
| mamba_mla_s2 | 0.97 | 0.93 | 0.89 | 0.87 |

> recall_eval MQAR / RULER-multikey are ~0 across all variants: content-based associative recall is
> not acquired by light adaptation of a small FineWeb checkpoint (only the fixed-token passkey induction
> is). The dense-recall frontier is therefore characterized in Track-B (from-scratch MQAR), not here.
