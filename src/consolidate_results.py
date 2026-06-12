"""Consolidate every run's RESULT_JSON (from the .map batch logs) into a single local ledger
experiments/results/results.jsonl — one JSON object per run, merging the input config (exp_id,
variant, task, hyperparams) with the returned metrics (by index, since Modal .map preserves order).
No Modal/network needed; reads logs/ + configs/ written during the run.
"""
import json
import os

HERE = os.path.dirname(__file__)
LOGS = os.path.join(HERE, "logs")
CFGS = os.path.join(HERE, "configs")
OUTDIR = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(OUTDIR, exist_ok=True)

# (config file, log file, group tag)
BATCHES = [
    ("pretrain_wave1.json", "pretrain_wave1.log", "trackA_pretrain"),
    ("pretrain_wave2.json", "pretrain_wave2.log", "trackA_pretrain"),
    ("B_h4_dc_sweep.json", "B_h4_dc_sweep.log", "trackB_h4_dc"),
    ("B_h4c_fixedffn.json", "B_h4c_fixedffn.log", "trackB_h4c_fixedffn"),
    ("B_h5_offload.json", "B_h5_offload.log", "trackB_h5_offload"),
    ("B_baselines.json", "B_baselines.log", "trackB_baselines"),
    ("B_a4_recall_source.json", "B_a4_recall_source.log", "trackB_a4"),
    ("eval_efficiency.json", "eval_efficiency.log", "eval_efficiency"),
    ("eval_recall.json", "eval_recall.log", "eval_recall"),
]


def read_result_json(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read().replace("\r", "\n")
    dec = json.JSONDecoder()
    best = None
    for line in txt.splitlines():
        i = line.find("RESULT_JSON:")
        if i >= 0:
            s = line[i + len("RESULT_JSON:"):].lstrip()
            try:
                obj, _ = dec.raw_decode(s)   # parse the JSON object, ignore any trailing junk
            except ValueError:
                continue
            if not isinstance(obj, dict) or "batch_results" not in obj:
                continue                      # skip non-result lines (strings/partials)
            # prefer the result with the most batch_results (a full run over a truncated one)
            if best is None or len(obj.get("batch_results", [])) >= len(best.get("batch_results", [])):
                best = obj
    return best


rows = []
for cfg_name, log_name, group in BATCHES:
    cfg_path = os.path.join(CFGS, cfg_name)
    log_path = os.path.join(LOGS, log_name)
    if not (os.path.exists(cfg_path) and os.path.exists(log_path)):
        print(f"  SKIP {cfg_name} (missing config or log)")
        continue
    cfg = json.load(open(cfg_path))
    inputs = cfg.get("batch_configs", [])
    res = read_result_json(log_path)
    if res is None:
        print(f"  WARN {log_name}: no RESULT_JSON")
        continue
    outputs = res.get("batch_results", [])
    if len(inputs) != len(outputs):
        print(f"  WARN {cfg_name}: {len(inputs)} inputs vs {len(outputs)} outputs (zipping by min)")
    for cin, cout in zip(inputs, outputs):
        if cout is None:
            rows.append({"group": group, "exp_id": cin.get("exp_id"), "status": "null_result", **cin})
            continue
        # merge: config first (exp_id/variant/hyperparams), then metrics (override/augment)
        row = {"group": group, "exp_id": cin.get("exp_id")}
        row.update({k: v for k, v in cin.items() if k != "batch_configs"})
        row.update(cout)
        rows.append(row)

out_path = os.path.join(OUTDIR, "results.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

# quick group summary
from collections import Counter
g = Counter(r["group"] for r in rows)
print(f"wrote {len(rows)} rows -> {out_path}")
for k, v in g.items():
    print(f"  {k}: {v}")
