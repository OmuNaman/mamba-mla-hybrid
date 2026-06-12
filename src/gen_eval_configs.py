"""Generate eval configs (recall_eval + efficiency) from the TRAINED Track-A checkpoints.
Run AFTER all 17 pretrains finish (each writes /vol/<exp_id>/model.pt = {state_dict, cfg}).

recall_eval: load a checkpoint, light equal-budget adapt (seed 11) -> eval MQAR-spread / passkey /
RULER-multikey on a DISJOINT split (seed 99) at lengths up to 4096 (eval_recall_acc uses a fixed
batch=64, so the full-attention variants would OOM past 4k; the long-context MEMORY story is carried
by `efficiency`, which is forward-only with a per-cell OOM-catch up to 64k).

efficiency: load a checkpoint, sweep (length x batch) -> directly-measured cache bytes, mamba-state
bytes, FLOPs/token, weights bytes, uncached decode tok/s. Architecture-level (one per arch + scale).
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "configs")
os.makedirs(OUT, exist_ok=True)


def w(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)
    return name


def ckpt(exp):
    return f"/vol/{exp}/model.pt"


SEED0 = ["transformer", "mamba2", "mamba_full", "mamba_swa", "mamba_mla", "transformer_mla"]
# 3-seed CIs for the 3 headline variants (seed 0 already in SEED0 above)
HEADLINE_SEEDS = [("mamba_full", 1), ("mamba_full", 2),
                  ("mamba_swa", 1), ("mamba_swa", 2),
                  ("mamba_mla", 1), ("mamba_mla", 2)]

# ---- recall_eval (trained-checkpoint recall; lengths <= 4096 to stay OOM-safe at batch=64) ----
recall = []
for v in SEED0:
    e = f"S_{v}_s0"
    recall.append({"task": "recall_eval", "exp_id": f"R_{e}", "ckpt": ckpt(e),
                   "lengths": [512, 1024, 2048, 4096], "n_kv": 16, "n_needles": 8,
                   "adapt_steps": 800, "adapt_len": 256, "adapt_lr": 0.001})
for v, s in HEADLINE_SEEDS:
    e = f"S_{v}_s{s}"
    recall.append({"task": "recall_eval", "exp_id": f"R_{e}", "ckpt": ckpt(e),
                   "lengths": [512, 1024, 2048, 4096], "n_kv": 16, "n_needles": 8,
                   "adapt_steps": 800, "adapt_len": 256, "adapt_lr": 0.001})
w("eval_recall.json", {"gpu": "A100-80GB", "batch_configs": recall})

# ---- efficiency (cache bytes / FLOPs / decode tok/s; arch-level: 6 size-S + 3 size-M + 2 A1 ratio) ----
eff_exps = [f"S_{v}_s0" for v in SEED0] \
    + [f"M_{v}_s0" for v in ("mamba_full", "mamba_swa", "mamba_mla")] \
    + ["A1_mla_r3", "A1_mla_r7"]
eff = []
for e in eff_exps:
    eff.append({"task": "efficiency", "exp_id": f"E_{e}", "ckpt": ckpt(e),
                "lengths": [4096, 16384, 65536], "batches": [1, 4],
                "require_kernel": True})   # throughput must be kernel-path; fail loud, never silent fallback
w("eval_efficiency.json", {"gpu": "A100-80GB", "batch_configs": eff})

print(f"wrote eval_recall.json ({len(recall)} runs) + eval_efficiency.json ({len(eff)} runs) -> {OUT}")
print(f"  recall: {[c['exp_id'] for c in recall]}")
print(f"  efficiency: {[c['exp_id'] for c in eff]}")
