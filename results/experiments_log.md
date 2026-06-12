# Experiments Log — mamba-mla-hybrid

## Code build + local verification (2026-06-09, before any Modal spend)
`experiments/modal/modal_app.py` implements the full study (one file, task-dispatched): faithful
**MLA with decoupled RoPE** (DeepSeek-V2 Eqs 9-19; cache = latent c^KV + shared k^R only), MHA/SWA,
**Mamba-2** (official `mamba_ssm.Mamba2` kernel with a correct **chunked-SSD pure-PyTorch fallback**
used uniformly if the kernel is absent), SwiGLU FFN, param-matching via FFN hidden size (depth &
d_model locked), direct cache-byte measurement, and leakage-free MQAR/RULER/passkey generators.

**Local GPU sanity (torch 2.6, no Modal):** caught + fixed two real bugs before any audit/Modal run:
1. chunked-SSD einsum reused subscript `n` for both chunk-index and state-dim -> rewrote with distinct
   subscripts (b,w,i,j,h,p,s).
2. intra-chunk `Lmat = exp(dAcum_i - dAcum_j)` overflowed to +inf in the masked (i<j) upper triangle,
   and inf*0 (tri mask) -> NaN for ALL Mamba variants -> fixed by masking the exponent to -inf BEFORE
   exp.
After the fixes: smoke passes (all 6 variants finite), param-match spread 1.7% (<3%), MQAR train/test
disjoint (no leakage), and the predicted science ordering already emerges at tiny scale
(transformer/MLA > pure-Mamba/SWA on dense recall). The pure-PyTorch SSD fallback is SLOW -> the
official mamba-ssm kernel matters for Track-A (2B-token) feasibility; verified at the Modal smoke.

## Modal verification + matrix launch (2026-06-10)

**Connectivity** restored after a local network SSL-inspection issue (gRPC/HTTP-2 was being broken by
a proxy/AV); resolved on the user's side. `modal app list` clean.

**Smoke (A100-80GB, free-tier dims):** `mamba_impl=kernel`, kernel available, all 6 variants finite,
`mqar_train_test_identical_any=false` (no leakage), cache-elem ordering sane
(transformer_mla 10 240 < mamba_mla 206 080 < mamba2 271 360 elems @ ctx 16).

**Precision probes (16 steps each, A100-80GB, mamba_mla, the costliest variant):**
| probe | dims | batch | s/step | tok/s | peak GB | non-embed params |
|---|---|---|---|---|---|---|
| size-S | d768 / 12L | 16 | 0.382 | 85 870 | 41.4 / 80 | 113.2 M |
| size-M | d1024 / 24L | 8 | 0.476 | 34 450 | 42.1 / 80 | 403.0 M |

Both confirm **kernel** (not the ~3x-slower fallback) and leave ~halve the GPU free. Wall-clock from
measured throughput: **size-S 1.5B tok ≈ 4.85 h**, **size-M 0.5B tok ≈ 4.0 h** — both << the 24 h
per-call timeout, and checkpoint-resume covers any preemption.

**Matrix launched (all `modal run --detach`, parallel):**
- `pretokenize` (A10G) → FineWeb-Edu 2.5B tokens to the Volume — **gates Track-A pretrains**.
- Track-B L4 fan-out (synthetic MQAR, no data dependency, running concurrently): `B_h4_dc_sweep` (72),
  `B_h4c_fixedffn` (36), `B_h5_offload` (54), `B_baselines` (60), `B_a4_recall_source` (9) = 231 L4 runs.
- On pretokenize completion: `pretrain_wave1` (6 size-S seed-0, parallel A100 `.map`) →
  `pretrain_wave2` (size-S extra seeds + size-M + A1 ratio) → recall_eval + efficiency evals.

### Pretokenize bottleneck caught + fixed live (2026-06-10)
The first pretokenize launch tokenized FineWeb-Edu **one document at a time** in a Python loop on an
otherwise-idle A10G (GPU unused — the task is pure CPU). Measured throughput **0.30 M tok/s** → ETA to
the 2.5B target **~122 min**, gating all Track-A pretrains for ~2 h. Fixed without touching the running
Track-B jobs (containers use their own mounted copy of `modal_app.py`, so a local edit is safe):
- **Batched tokenization**: accumulate `tok_batch=2048` documents and call the HF *fast* (Rust)
  tokenizer on the whole list at once → it parallelizes across cores via rayon (vs the per-doc Python
  loop). Same packing semantics (each doc's ids + EOT, uint16, capped at target).
- **16-core container**: added a `cpu` override in the entrypoint; relaunched pretokenize with
  `cpu=16` (gives the Rust tokenizer real parallelism) and target trimmed to **2.0B** (1.5B train +
  50M val holdout + margin — covers every run; size-S needs ≤1.55B, size-M ≤0.55B).
Stopped the slow run (`modal app stop -y`), confirmed its container died, relaunched with `force=true`
to overwrite the partial bin. Net: discarded ~0.3B of throwaway tokens to save ~90 min on the critical
path. (New rate confirmed at relaunch.)

### Run 2026-06-11 — matrix executing
- **Pretokenize (batched/16-core):** confirmed **~3.5 M tok/s** (11.6× the old 0.30); produced exactly
  **2.0B FineWeb-Edu** train tokens + **252 472** WikiText-103 val tokens (both bins + meta + metrics on
  the Volume). ~12 min wall.
- **Track-B: 231/231 MQAR runs COMPLETE** (all 5 sweeps: H4 d_c, H4 fixed-FFN, H5 offload, baselines,
  A4). Frontier signal sane out of the box, e.g. `mamba_mla` d_c=16/nkv=32 → mqar_acc≈0.0002 (recall
  collapses at extreme compression).
- **Wave 1 (6 size-S, A100, seed 0): healthy.** All 6 past warmup with uniform steep descent
  (init 709/360 → ~48–51 at step 57/2861); zero crashes/OOM, `mamba_impl=kernel` on all. ~2861 optim
  steps/run.
- **Wave 2 (11 runs: seeds 1/2 ×3 headline + 3 size-M + 2 A1 ratio) launched** on the overlap strategy
  once Wave-1 health was confirmed; Modal schedules them as A100 slots free.

### CRITICAL BUG found in consolidation + fixed (2026-06-11) — Track-B MQAR re-run
Aggregating the first full result set exposed that **Track-B MQAR was ~0 for ALL 231 runs** (max 0.018),
including the pure **transformer** — which should ace MQAR. When the easy baseline fails, the task/metric
is broken, not the architectures. An instrumented run pinned the root cause:
- **train_acc 0.97 / test_acc 0.017 = MEMORIZATION.** The `mqar` task built a FIXED pool of 20 000 train
  sequences and sampled from it, so the small model memorized those specific sequences instead of learning
  the in-context retrieval ALGORITHM, and failed on the disjoint (seed+777) test pool. (The code-audit had
  checked "shortcuts floor at chance" and "train/test disjoint" but NOT "is the pool large enough to
  prevent memorization" — the missing learnability check.)
**Fixes (re-validated live):**
1. **Fresh random MQAR every step** (effectively infinite data) in both the `mqar` task and `recall_eval`
   adaptation -> the model must learn, not memorize. Capability sweep after the fix: loss -> 0, test acc
   1.00 (was ~0).
2. **Calibrated step budget 3 000 -> 12 000.** The old 3 000 was tuned to the memorization regime; learning
   the real algorithm on fresh data needs ~10-20k steps (n_kv=8 solves ~6k; n_kv=16 ~0.94 at 20k). n_kv=8
   is solved even by pure Mamba -> the discriminating regime (where the d_c frontier lives) is **n_kv>=16**.
3. **Perf:** vectorized the dense-MQAR generator (argpartition over a capped key space) and **gathered the
   sparse supervised positions before the float-cast/cross-entropy** (MQAR supervises ~nq of 512 tokens;
   the old code ran CE over all 268M logits/step). ~152 -> 114 ms/step (residual is launch-bound on L4).
**Re-run launched:** all 5 Track-B sweeps (231 runs) at 12 000 steps with the fix; ~2 h on L4.
NOTE: the other result legs are UNAFFECTED and stand — efficiency cache Pareto (MLA 5-30x < full-attn),
pretrain PPL (3-seed), and recall_eval passkey extrapolation (mamba_mla holds 0.98 at 4k vs transformer 0.06).

### Track-B re-run LAUNCHED 2026-06-11 21:28 IST (with the memorization fix)
All 5 sweeps re-launched (231 runs @ 12 000 steps, fresh-data MQAR + gather-loss + vectorized gen).
App IDs in `logs/_rerun_apps.txt`. ~50 L4 concurrent, ETA ~2 h (~23 min/run × ~5 waves). Early health
at 21:3x: first wave ~step 1800/12000, `B_bl_transformer_nkv16_s0` loss 9.0→5.4 (LEARNING ✓), no errors.
On completion: `consolidate_results.py` + `aggregate.py` (overwrites the broken Track-B rows) → sanity-check
the frontier (easy baselines high MQAR, d_c knee, mamba2/swa fall off at high n_kv) → experiment-review gate.

### Track-B re-run RESULT (2026-06-12, after the memorization fix) — headline d_c frontier does NOT hold
Re-run completed for the two decisive sweeps (d_c sweep 72 + baselines 60); h5/h4c/a4 stopped partway
(confirmatory, not needed — same pattern). The fix worked mechanically (models now LEARN: transformer
n_kv=16 test 1.0 vs old 0.017), **but the intended H4 d_c-vs-recall knee + H5 Mamba-offload do NOT
materialize at this scale (d_model=256, seq_len=512):**
- **d_c sweep (mamba_mla), mean acc, rows=d_c cols=n_kv:** nkv16 all 1.0; nkv32 ~1.0; nkv64 scatter
  0.67–1.0; nkv128 noise 0.00–1.0 (d_c=16→0.33 even BEATS d_c=256→0.00). No monotonic knee in any column.
- **Cause:** MLA caches c^KV PER TOKEN (d_c × seq_len), so at seq 512 even d_c=16 → ~8k cache dims =
  ample for ≤64 pairs; latent compression is NOT the binding constraint at this length → flat frontier.
  n_kv=128 is undertrained in 12k steps → pure seed-variance noise (and mamba2 0.86 > mamba_full 0.11 at
  nkv128 is impossible as real capability → confirms it's noise).
- **baselines:** nkv≤64 solved by EVERY variant incl. pure mamba2 (state holds ≤64 pairs at this scale)
  → no recurrent-vs-attention separation; nkv128 noisy.

**SOLID & UNAFFECTED (the real paper, if we pivot):** efficiency cache Pareto (MLA 5–30× < full-attn,
measured), pretrain PPL (3-seed, mamba_mla competitive/best), recall_eval passkey extrapolation
(mamba_mla 0.98 @4k vs transformer 0.06), and MLA recall **robust to aggressive d_c compression**
(reinforces the efficiency story). DECISION PENDING (user): B=pivot to this story + report no-knee as a
finding; C=pivot + one harder-regime d_c sweep (seq 2–4k, d_c{4,8}, more steps); A=full re-run chasing knee.

Consolidated ledger: `experiments/results/results.jsonl` (172 valid runs) + `results_summary.md`.

### Experiment-Review GATE (2026-06-12) — PASS (conditional on a claims-contract; NO re-runs)
Ran the results-validity gate on the final ledger: independent **Gemini 3.1 Pro (grounded)** + 3
adversarial Claude personas (Leakage Auditor, Skeptical Empiricist, Baseline Advocate). Verdict
**3× REVISE + 1× PASS → resolves to PASS-with-mandatory-scoping**, because **all four agree every
required action is a paper-scoping / claim-hedging change — none requires a new experiment.** The gate
certifies the *results* are valid, fair, and fully traceable; what must change is the *claims*.
- **Clean (unanimous):** no leakage (fresh-data-per-step + disjoint test seed + held-out val all verified
  in code; `train_acc ≈ mqar_acc` to ~0.001 across all 132 Track-B rows = memorization bug definitively
  gone); baselines fair (param_match_err <0.27%, transformer carries the *largest* FFN, one Mamba-2
  kernel, shared tokenizer/data/steps/optimizer); cache 5–30× verifies to the byte; every headline
  number traces to a real row; no too-good-to-be-true win.
- **No-knee is a REAL null (unanimous), reported as a LIMITATION not a result:** MLA caches c^KV per
  token → at seq 512 even d_c=16 (~8k latent dims) is non-binding for ≤64 pairs; n_kv=128 is undertrained
  (the impossible `mamba2 0.86 > mamba_full 0.11` confirms seed noise). It is "H4/H5 not stressed at this
  scale," NOT "no knee exists."
- **Claims-contract carried into plan-paper (binding):** LEAD with S1 (measured cache Pareto + SWA↔MLA
  crossover) and S2 (size-S 3-seed PPL equivalence, Welch n.s.); SOFTEN S3 passkey to qualitative (drop
  "0.98 vs 0.06"; transformer is n=1, collapse likely RoPE extrapolation; fixed the adapt-len-256 label
  in results_summary.md); DROP H4 d_c-knee + H5 offload as findings → honest null + mechanism; FOLD S4
  "d_c-robustness" into the null (it's saturation, not robustness); DOWNGRADE all size-M + n=1 references
  to single-seed points; HEDGE the pure-transformer PPL gap (single shared LR → attention baselines
  likely under-tuned); add a Limitations section.
Full synthesis: `critique/experiment_review.md`; machine verdict: `critique/experiment_verdict.json`.
**Decision: proceed to plan-paper (Step 4) under the claims-contract. No further GPU runs.**

(Per-run RESULT_JSON records appended below as each batch completes.)
