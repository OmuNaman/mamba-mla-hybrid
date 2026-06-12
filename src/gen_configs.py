"""Generate Modal config JSONs for the experiment matrix into ./configs/.
Track A (pretrain on FineWeb-Edu, A100-80GB) + Track B (synthetic MQAR sweep, L4 fan-out).
recall_eval / efficiency configs are written by gen_eval_configs.py AFTER pretrain (need ckpt paths).
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "configs")
os.makedirs(OUT, exist_ok=True)


def w(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)
    return name


VARIANTS = ["transformer", "mamba2", "mamba_full", "mamba_swa", "mamba_mla", "transformer_mla"]
HEADLINE = ["mamba_full", "mamba_swa", "mamba_mla"]   # extra seeds + size M
files = []

# ---- pretokenize (ONCE) ----
files.append(w("pretokenize.json", {"task": "pretokenize", "exp_id": "pretokenize",
                                     "gpu": "A10G", "target_tokens": 2_500_000_000}))

# ---- Track A size S (~125M): d_model 768, 12 layers ----
# Grounded by the kernel probe: mamba_mla ~80K tok/s @ batch8/22GB -> batch16 fits (~44GB) & faster;
# 1.5B tokens is ample to fix the PPL ordering + cache/recall story at this controlled scale.
S = dict(d_model=768, n_layers=12, n_heads=12, d_head=64, vocab=50304, seq_len=2048,
         swa_window=512, attn_ratio=5, ref_d_ff=3072, d_c=256, d_cq=384, d_rope=32,
         tokens=1_500_000_000, lr=1.2e-3, batch=16, accum=16, bf16=True, gpu="A100-80GB",
         require_kernel=True)   # fail fast rather than silently run the ~3x-slower fallback
pre_s0, pre_seeds, pre_m, pre_a1 = [], [], [], []   # collected for parallel .map batch files
for v in VARIANTS:
    c = {"task": "pretrain", "exp_id": f"S_{v}_s0", "variant": v, "seed": 0, **S}
    files.append(w(f"S_{v}_s0.json", c)); pre_s0.append(c)
# extra seeds on the headline variants (for TOST + significance)
for v in HEADLINE:
    for s in (1, 2):
        c = {"task": "pretrain", "exp_id": f"S_{v}_s{s}", "variant": v, "seed": s, **S}
        files.append(w(f"S_{v}_s{s}.json", c)); pre_seeds.append(c)

# ---- Track A size M (~350M): d_model 1024, 24 layers (scale-stability) ----
# batch 8 @ d=1024/24L stays well under 80GB; 1B tokens for the second scale point.
M = dict(d_model=1024, n_layers=24, n_heads=16, d_head=64, vocab=50304, seq_len=2048,
         swa_window=512, attn_ratio=5, ref_d_ff=4096, d_c=256, d_cq=384, d_rope=32,
         tokens=500_000_000, lr=6e-4, batch=8, accum=16, bf16=True, gpu="A100-80GB",
         require_kernel=True)
for v in HEADLINE:   # 3 headline variants at the second scale (scale-stability check)
    c = {"task": "pretrain", "exp_id": f"M_{v}_s0", "variant": v, "seed": 0, **M}
    files.append(w(f"M_{v}_s0.json", c)); pre_m.append(c)

# ---- A1 ratio ablation (size S, mamba_mla @ 1:3 / 1:5 / 1:7) ----
for r in (3, 7):   # r=5 is the default already covered by S_mamba_mla_s0
    cc = dict(S); cc["attn_ratio"] = r
    c = {"task": "pretrain", "exp_id": f"A1_mla_r{r}", "variant": "mamba_mla", "seed": 0, **cc}
    files.append(w(f"A1_mla_r{r}.json", c)); pre_a1.append(c)

# =========================================================================
# Track B — synthetic MQAR (small models, L4 fan-out via batch_configs)
# small model shared base; vocab 8192 (Zoology); steps 3000
# =========================================================================
TB = dict(d_model=256, n_layers=4, n_heads=4, d_head=64, vocab=8192, d_cq=384, d_rope=32,
          swa_window=64, attn_ratio=3, steps=12000, batch=64, lr=1e-3, n_train=20000, bf16=True)
DC_GRID = [16, 32, 64, 128, 256, 512]
NKV_GRID = [16, 32, 64, 128]
SEEDS = [0, 1, 2]
LEN = 512

# H4: d_c sweep for mamba_mla across MQAR difficulty (the headline frontier; param-matched via FFN)
h4 = []
for dc in DC_GRID:
    for nkv in NKV_GRID:
        for sd in SEEDS:
            h4.append({"task": "mqar", "exp_id": f"B_h4_dc{dc}_nkv{nkv}_s{sd}", "variant": "mamba_mla",
                       "seed": sd, "seq_len": LEN, "n_kv": nkv, "d_c": dc, **TB})
files.append(w("B_h4_dc_sweep.json", {"gpu": "L4", "batch_configs": h4}))

# H4 fixed-FFN CONTROL (code-audit rigor): hold d_ff constant across d_c (match_params=False) so the
# recall-vs-d_c knee is NOT confounded by FFN compensation. d_c* reported under BOTH regimes.
h4c = []
for dc in DC_GRID:
    for nkv in (32, 64):
        for sd in SEEDS:
            c = dict(TB); c["match_params"] = False; c["d_ff"] = 1024  # fixed 4*d_model
            h4c.append({"task": "mqar", "exp_id": f"B_h4c_dc{dc}_nkv{nkv}_s{sd}", "variant": "mamba_mla",
                        "seed": sd, "seq_len": LEN, "n_kv": nkv, "d_c": dc, **c})
files.append(w("B_h4c_fixedffn.json", {"gpu": "L4", "batch_configs": h4c}))

# H5: Mamba-backbone vs SWA-backbone vs full-backbone, each + 1 MLA layer, d_c sweep, nkv=64 (hard)
h5 = []
for bb in ["mamba_mla", "swa_mla", "transformer_mla"]:
    for dc in DC_GRID:
        for sd in SEEDS:
            h5.append({"task": "mqar", "exp_id": f"B_h5_{bb}_dc{dc}_s{sd}", "variant": bb,
                       "seed": sd, "seq_len": LEN, "n_kv": 64, "d_c": dc, **TB})
files.append(w("B_h5_offload.json", {"gpu": "L4", "batch_configs": h5}))

# baselines on MQAR: show SWA / pure-Mamba fail dense MQAR; full / MLA hold
bl = []
for v in ["transformer", "mamba2", "mamba_swa", "mamba_full", "mamba_mla"]:
    for nkv in NKV_GRID:
        for sd in SEEDS:
            bl.append({"task": "mqar", "exp_id": f"B_bl_{v}_nkv{nkv}_s{sd}", "variant": v,
                       "seed": sd, "seq_len": LEN, "n_kv": nkv, "d_c": 256, **TB})
files.append(w("B_baselines.json", {"gpu": "L4", "batch_configs": bl}))

# A4 recall-source: remove-MLA (pure mamba2) vs remove-Mamba (transformer_mla) vs hybrid, nkv=64
a4 = []
for v in ["mamba_mla", "mamba2", "transformer_mla"]:
    for sd in SEEDS:
        a4.append({"task": "mqar", "exp_id": f"B_a4_{v}_s{sd}", "variant": v, "seed": sd,
                   "seq_len": LEN, "n_kv": 64, "d_c": 256, **TB})
files.append(w("B_a4_recall_source.json", {"gpu": "L4", "batch_configs": a4}))

# ---- parallel fan-out batch files (Modal .map runs them concurrently up to the GPU quota) ----
# Wave 1: the 6 size-S seed-0 variants (the core matched comparison -> all variants' PPL/cache fastest).
files.append(w("pretrain_wave1.json", {"gpu": "A100-80GB", "batch_configs": pre_s0}))
# Wave 2: size-S extra seeds + size-M scale check + A1 ratio ablation.
files.append(w("pretrain_wave2.json", {"gpu": "A100-80GB", "batch_configs": pre_seeds + pre_m + pre_a1}))

print(f"wrote {len(files)} config files to {OUT}")
print(f"  Wave 1 (parallel): {len(pre_s0)} runs | Wave 2 (parallel): {len(pre_seeds)+len(pre_m)+len(pre_a1)} runs")
print(f"  Track B (L4 fan-out): B_h4_dc_sweep, B_h4c_fixedffn, B_h5_offload, B_baselines, B_a4_recall_source")
