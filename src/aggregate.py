"""Aggregate results.jsonl into the headline summary tables (-> experiments/results/results_summary.md):
pretrain PPL (mean+/-std over seeds), Track-B MQAR baselines + d_c frontier, efficiency cache vs length,
recall passkey vs length. Pure stdlib so it runs anywhere.
"""
import json
import os
import statistics as st
from collections import defaultdict

RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
rows = [json.loads(l) for l in open(os.path.join(RES, "results.jsonl"), encoding="utf-8")]


def g(group):
    return [r for r in rows if r.get("group") == group and r.get("status") != "null_result"]


out = []
def w(s=""):
    out.append(s)

# ---------- Track-A pretrain PPL ----------
w("# Results Summary — mamba-mla-hybrid\n")
w("## Track-A pretrain perplexity (FineWeb-Edu val / WikiText-103 val)\n")
pre = g("trackA_pretrain")
by = defaultdict(lambda: defaultdict(list))   # (size,variant) -> metric -> [vals over seeds]
for r in pre:
    exp = r.get("exp_id", "")
    size = "M" if exp.startswith("M_") else ("A1:r%s" % r.get("attn_ratio") if exp.startswith("A1_") else "S")
    var = r.get("variant", "?")
    key = (size, var, exp.split("_s")[0] if size != "A1" else exp)
    by[(size, var)]["ppl_fw"].append(r.get("ppl_fineweb"))
    by[(size, var)]["ppl_wt"].append(r.get("ppl_wikitext"))
w("| size | variant | ppl_fineweb (mean±std, n) | ppl_wikitext |")
w("|---|---|---|---|")
def fmt(vs):
    vs = [v for v in vs if isinstance(v, (int, float))]
    if not vs:
        return "-"
    if len(vs) == 1:
        return f"{vs[0]:.2f} (n=1)"
    return f"{st.mean(vs):.2f}±{st.pstdev(vs):.2f} (n={len(vs)})"
for (size, var) in sorted(by):
    m = by[(size, var)]
    w(f"| {size} | {var} | {fmt(m['ppl_fw'])} | {fmt(m['ppl_wt'])} |")

# A1 ratio ablation explicit
w("\n### A1 ratio ablation (mamba_mla, size-S)\n")
w("| run | attn_ratio | ppl_fineweb |")
w("|---|---|---|")
for r in pre:
    if str(r.get("exp_id", "")).startswith("A1_"):
        w(f"| {r['exp_id']} | {r.get('attn_ratio')} | {r.get('ppl_fineweb'):.2f} |")

# ---------- Track-B MQAR baselines (H3: mamba_mla > mamba_swa) ----------
w("\n## Track-B MQAR baselines — dense recall accuracy by variant (mean over n_kv, seeds)\n")
bl = g("trackB_baselines")
acc = defaultdict(list)
for r in bl:
    acc[r.get("variant", "?")].append(r.get("mqar_acc"))
w("| variant | MQAR acc (mean±std, n) |")
w("|---|---|")
for v in ["transformer", "mamba2", "mamba_swa", "mamba_full", "mamba_mla"]:
    if v in acc:
        w(f"| {v} | {fmt(acc[v])} |")

# ---------- Track-B H4 d_c frontier (mamba_mla MQAR acc vs d_c) ----------
w("\n## Track-B H4 — MQAR accuracy vs MLA latent dim d_c (mamba_mla; mean over n_kv, seeds)\n")
h4 = g("trackB_h4_dc")
byd = defaultdict(list)
for r in h4:
    byd[r.get("d_c")].append(r.get("mqar_acc"))
w("| d_c | MQAR acc (mean±std, n) |")
w("|---|---|")
for dc in sorted(byd, key=lambda x: (x is None, x)):
    w(f"| {dc} | {fmt(byd[dc])} |")

# ---------- Efficiency cache vs length ----------
w("\n## Efficiency — KV/latent cache bytes per sequence vs context length\n")
eff = g("eval_efficiency")
order = ["transformer", "mamba2", "mamba_full", "mamba_swa", "mamba_mla", "transformer_mla"]
w("| variant(size) | cache@4k (MB) | @16k | @64k | mamba_state(MB) |")
w("|---|---|---|---|---|")
for r in eff:
    exp = r.get("exp_id", "")
    tag = exp.replace("E_", "")
    by_len = {x["length"]: x for x in r.get("by_length", [])}
    def mb(L):
        v = by_len.get(L, {}).get("cache_bytes_per_seq")
        return f"{v/1e6:.0f}" if isinstance(v, (int, float)) else "-"
    ms = by_len.get(4096, {}).get("mamba_state_bytes_per_seq")
    w(f"| {tag} | {mb(4096)} | {mb(16384)} | {mb(65536)} | {ms/1e6:.0f} |" if isinstance(ms,(int,float)) else f"| {tag} | {mb(4096)} | {mb(16384)} | {mb(65536)} | - |")

# ---------- Recall passkey vs length ----------
w("\n## recall_eval — passkey accuracy vs length (trained-checkpoint extrapolation; train len = 2048)\n")
rc = g("eval_recall")
w("| variant | 512 | 1024 | 2048 | 4096 |")
w("|---|---|---|---|---|")
for r in rc:
    pk = r.get("by_task", {}).get("passkey", {})
    w(f"| {r.get('exp_id','').replace('R_S_','')} | " + " | ".join(f"{pk.get(str(L),0):.2f}" for L in [512,1024,2048,4096]) + " |")
w("\n> recall_eval MQAR / RULER-multikey are ~0 across all variants: content-based associative recall is")
w("> not acquired by light adaptation of a small FineWeb checkpoint (only the fixed-token passkey induction")
w("> is). The dense-recall frontier is therefore characterized in Track-B (from-scratch MQAR), not here.")

path = os.path.join(RES, "results_summary.md")
open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"wrote {path}")
print("\n".join(out))
