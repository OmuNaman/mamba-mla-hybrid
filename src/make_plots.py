"""Generate the 5 data plots (Fig 1,3,4,5,6) directly from experiments/results/results.jsonl.
Exact numbers only (no invented data); writes CSVs to data/ for traceability and PNGs to output/.
Honors the experiment-review claims-contract: no trend line through the n_kv=64/128 MQAR noise;
passkey shown qualitatively with the transformer marked n=1; sample std (ddof=1) on the 3-seed arms.
Cache MB = bytes / 1e6 (decimal MB, matching results_summary.md / the paper)."""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LEDGER = os.path.join(ROOT, "experiments", "results", "results.jsonl")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open(LEDGER, encoding="utf-8")]
def grp(g): return [r for r in rows if r.get("group") == g]

# ---- consistent styling ----
# Fonts sized so the figures stay legible when scaled to IEEE single-column width (~3.3in).
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "axes.titlesize": 14.5,
    "axes.labelsize": 14, "legend.fontsize": 11.5, "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "axes.grid": True,
    "grid.alpha": 0.30, "grid.linewidth": 0.6, "axes.axisbelow": True,
})
# variant -> (label, color, marker); mamba_mla = proposed (bold purple)
VAR = {
    "transformer":     ("Transformer (full attn)",   "#c0504d", "s", 1.6, "--"),
    "transformer_mla": ("Transformer + MLA",          "#e1a06a", "v", 1.6, "--"),
    "mamba2":          ("Mamba-2 (pure)",             "#7f7f7f", "D", 1.6, "-."),
    "mamba_full":      ("Mamba-2 + full (Jamba)",     "#4f81bd", "o", 1.8, "-"),
    "mamba_swa":       ("Mamba-2 + SWA (Samba)",      "#9bbb59", "^", 1.8, "-"),
    "mamba_mla":       ("Mamba-2 + MLA (ours)",       "#7030a0", "*", 2.6, "-"),
}
ORDER = ["transformer", "transformer_mla", "mamba2", "mamba_full", "mamba_swa", "mamba_mla"]
MB = 1e6


def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print("wrote", p)


def write_csv(name, header, data):
    p = os.path.join(DATA, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(data)
    print("wrote", p)


# ===================== FIG 1 — decode cache vs context length (HERO) =====================
eff = grp("eval_efficiency")
def eff_var(r):  # E_S_<variant>_s0  -> variant
    e = r["exp_id"]
    for v in ORDER:
        if e == f"E_S_{v}_s0": return v
    return None
LENS = [4096, 16384, 65536]
fig1_rows = []
fig, ax = plt.subplots(figsize=(6.4, 4.9))
for v in ORDER:
    r = next((x for x in eff if eff_var(x) == v), None)
    if r is None: continue
    bl = {d["length"]: d["cache_bytes_per_seq"] / MB for d in r["by_length"]}
    ys = [bl[L] for L in LENS]
    lab, col, mk, lw, ls = VAR[v]
    ax.plot(LENS, ys, marker=mk, color=col, lw=lw, ls=ls, ms=9 if v == "mamba_mla" else 6,
            label=lab, zorder=5 if v == "mamba_mla" else 3)
    for L in LENS: fig1_rows.append([v, L, round(bl[L], 3)])
# SWA<->MLA crossover (per-seq, length-axis): t* = 2*n_h*d_h*w / (d_c+d_h^R)
tstar = 2 * 12 * 64 * 512 / (256 + 32)  # ~2730 tokens
ax.axvline(tstar, color="#999999", ls=":", lw=1.3, zorder=1)
ax.text(tstar * 1.07, 5.2, "SWA↔MLA crossover", rotation=90,
        fontsize=9.5, color="#666666", va="bottom", ha="left")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(LENS); ax.set_xticklabels(["4k", "16k", "64k"])
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Decode cache per sequence (MB)")
ax.set_title("Directly-measured decode cache vs context length (size-S, 125M)\n"
             "MLA hybrid: 5–30× below full attention; SWA bounded; Mamba-2 constant", fontsize=11.5)
ax.legend(loc="upper left", framealpha=0.95, ncol=1)
savefig(fig, "fig1_cache_vs_length.png")
write_csv("fig1_cache_vs_length.csv", ["variant", "length", "cache_mb"], fig1_rows)


# ===================== FIG 3 — quality–cache Pareto (size-S) =====================
ta = grp("trackA_pretrain")
def s_rows(v): return [r for r in ta if r["exp_id"].startswith("S_") and r["variant"] == v]
fig3_rows = []
fig, ax = plt.subplots(figsize=(6.4, 4.9))
for v in ORDER:
    rs = s_rows(v)
    if not rs: continue
    ppls = np.array([r["ppl_fineweb"] for r in rs])
    cache_mb = rs[0]["cache_bytes_16k"] / MB
    mean = float(ppls.mean()); std = float(ppls.std(ddof=1)) if len(ppls) > 1 else 0.0
    lab, col, mk, lw, ls = VAR[v]
    ax.errorbar(cache_mb, mean, yerr=(std if len(ppls) > 1 else None), fmt=mk, color=col,
                ms=16 if v == "mamba_mla" else 10, capsize=4, elinewidth=1.4,
                mec="black" if v == "mamba_mla" else col, mew=1.2 if v == "mamba_mla" else 0,
                label=f"{lab}" + (f"  (n={len(ppls)})" if len(ppls) > 1 else "  (n=1)"), zorder=5)
    fig3_rows.append([v, round(mean, 2), round(std, 2), round(cache_mb, 2), len(ppls)])
ax.set_xscale("log")
ax.set_xlabel("Decode cache @16k context (MB, log)")
ax.set_ylabel("FineWeb-Edu perplexity (lower better)")
ax.set_title("Quality–cache frontier (size-S; 3-seed arms show ±1 SD)")
# annotate the tied hybrid cluster
# (no in-plot annotation; the caption conveys the tied-quality / big-cache-spread message)
ax.legend(loc="center right", framealpha=0.95)
ax.invert_yaxis()  # better (lower ppl) toward top
savefig(fig, "fig3_quality_cache_pareto.png")
write_csv("fig3_quality_cache_pareto.csv",
          ["variant", "ppl_fineweb_mean", "ppl_fineweb_std_ddof1", "cache_mb_16k", "n_seeds"], fig3_rows)


# ===================== FIG 4 — the d_c-vs-recall NULL (small multiples) =====================
dc = grp("trackB_h4_dc")  # mamba_mla only
DCS = [16, 32, 64, 128, 256, 512]
NKVS = [16, 32, 64, 128]
fig4_rows = []
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
# NO connecting lines anywhere — a line through per-seed means would imply a (false) d_c trend.
# Scatter-only + a flat reference at 1.0 in the (near-)saturated panels.
ANNOT = {16: ("saturated:\nall seeds ≈ 1.0", "#2a7a2a"),
         32: ("saturated\n(one seed dropout)", "#2a7a2a"),
         64: ("no monotonic\n$d_c$→recall trend", "#b03030"),
         128: ("undertrained:\nper-seed noise", "#b03030")}
for ax, nkv in zip(axes, NKVS):
    saturated = nkv in (16, 32)
    for d in DCS:
        accs = [r["mqar_acc"] for r in dc if r["d_c"] == d and r["n_kv"] == nkv]
        for r in dc:
            if r["d_c"] == d and r["n_kv"] == nkv:
                fig4_rows.append([nkv, d, r["seed"], round(r["mqar_acc"], 4), round(r["train_acc"], 4)])
        ax.scatter([d] * len(accs), accs, s=46, color="#7030a0", alpha=0.6,
                   edgecolor="black", linewidth=0.4, zorder=4)
    if saturated:
        ax.axhline(1.0, color="#888888", ls="--", lw=1.0, alpha=0.7, zorder=2)
    txt, col = ANNOT[nkv]
    ypos = 0.07 if saturated else 0.52
    ax.text(0.5, ypos, txt, transform=ax.transAxes, ha="center", fontsize=11, color=col)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DCS); ax.set_xticklabels([str(d) for d in DCS], fontsize=10)
    ax.set_ylim(-0.05, 1.08); ax.set_xlabel("MLA latent dim $d_c$")
    ax.set_title(f"# KV pairs = {nkv}")
axes[0].set_ylabel("MQAR accuracy (test)")
fig.suptitle("No minimum-latent ($d_c^*$) knee for dense recall at this scale  "
             "(mamba_mla, seq len 512; per-seed points, 3 seeds)", y=1.04, fontsize=14)
savefig(fig, "fig4_dc_null.png")
write_csv("fig4_dc_null.csv", ["n_kv", "d_c", "seed", "mqar_acc", "train_acc"], fig4_rows)


# ===================== FIG 5 — passkey length-extrapolation =====================
rc = grp("eval_recall")
PLENS = [512, 1024, 2048, 4096]
def passkey_series(v):
    out = {}
    for L in PLENS:
        vals = []
        for r in rc:
            if r["exp_id"].startswith(f"R_S_{v}_s"):
                pk = (r.get("by_task") or {}).get("passkey", {})
                if str(L) in pk: vals.append(pk[str(L)])
        out[L] = vals
    return out
fig5_rows = []
fig, ax = plt.subplots(figsize=(6.6, 5.0))
SHORT5 = {"transformer": "Transf", "transformer_mla": "Transf+MLA", "mamba2": "Mamba2",
          "mamba_full": "M2+full", "mamba_swa": "M2+SWA", "mamba_mla": "M2+MLA"}
for v in ORDER:
    ser = passkey_series(v)
    if not any(ser.values()): continue
    means = [np.mean(ser[L]) for L in PLENS]
    n = max(len(ser[L]) for L in PLENS)
    lab, col, mk, lw, ls = VAR[v]
    lab = SHORT5[v]
    if n > 1:  # 3-seed hybrids: mean + min/max band
        lo = [min(ser[L]) for L in PLENS]; hi = [max(ser[L]) for L in PLENS]
        ax.fill_between(PLENS, lo, hi, color=col, alpha=0.15, zorder=2)
        ax.plot(PLENS, means, marker=mk, color=col, lw=lw, ls="-",
                ms=11 if v == "mamba_mla" else 6, label=f"{lab} (n={n})", zorder=5)
    else:
        ax.plot(PLENS, means, marker=mk, color=col, lw=1.6, ls="--",
                ms=6, label=f"{lab} (n=1)", zorder=4)
    for L in PLENS:
        for s in ser[L]: fig5_rows.append([v, L, round(s, 4)])
ax.axvline(2048, color="#999999", ls=":", lw=1.3)
ax.annotate("pretraining context (2048)", xy=(2048, 0.08), xytext=(2048 * 0.55, 0.08),
            fontsize=10.5, color="#555555")
ax.set_xscale("log", base=2); ax.set_xticks(PLENS); ax.set_xticklabels(["512", "1k", "2k", "4k"])
ax.set_xlabel("Evaluation length (tokens)"); ax.set_ylabel("Passkey accuracy")
ax.set_ylim(-0.03, 1.05)
ax.set_title("Passkey extrapolation: Mamba-MLA holds, pure dense Transformer degrades\n"
             "(qualitative; references are single-seed)")
# legend below the axes so it never overlaps the curves
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.95, fontsize=11)
savefig(fig, "fig5_passkey_extrapolation.png")
write_csv("fig5_passkey_extrapolation.csv", ["variant", "length", "passkey_acc"], fig5_rows)


# ===================== FIG 6 — MQAR baseline heatmap =====================
bl = grp("trackB_baselines")
BVARS = ["transformer", "mamba_full", "mamba_mla", "mamba_swa", "mamba2"]
M = np.full((len(BVARS), len(NKVS)), np.nan)
fig6_rows = []
for i, v in enumerate(BVARS):
    for j, nkv in enumerate(NKVS):
        accs = [r["mqar_acc"] for r in bl if r["variant"] == v and r["n_kv"] == nkv]
        if accs:
            M[i, j] = float(np.mean(accs))
            fig6_rows.append([v, nkv, round(M[i, j], 4), len(accs)])
fig, ax = plt.subplots(figsize=(6.4, 4.8))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(NKVS))); ax.set_xticklabels([f"{n}" for n in NKVS])
ax.set_yticks(range(len(BVARS)))
SHORT = {"transformer": "Transformer", "mamba_full": "Mamba2 + full", "mamba_mla": "Mamba2 + MLA",
         "mamba_swa": "Mamba2 + SWA", "mamba2": "Mamba2 (pure)"}
ax.set_yticklabels([SHORT[v] for v in BVARS])
ax.set_xlabel("# KV pairs (recall density)"); ax.set_title("Track-B MQAR accuracy by variant (mean of 3 seeds)")
for i in range(len(BVARS)):
    for j in range(len(NKVS)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12.5,
                    color="black" if 0.25 < M[i, j] < 0.85 else "white")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("MQAR accuracy")
savefig(fig, "fig6_mqar_heatmap.png")
write_csv("fig6_mqar_heatmap.csv", ["variant", "n_kv", "mean_mqar_acc", "n_seeds"], fig6_rows)

print("\nALL PLOTS DONE")
