# Compressed-Latent vs. Sliding-Window vs. Full Attention as Mamba-2's Partner

A controlled, from-scratch, matched-budget study that swaps a Mamba-2 backbone's **attention partner**
among **full attention**, **sliding-window attention (SWA)**, and **DeepSeek compressed-latent MLA**
(with decoupled RoPE), at 125M and 350M parameters, and directly measures decode-cache bytes,
perplexity, and synthetic dense recall. The contribution is a rigorous, fully traceable characterization
plus a mechanistically-explained negative result, not a state-of-the-art or first-hybrid claim.

📄 **Paper:** [paper/paper.pdf](paper/paper.pdf) · 💻 reproducible on [Modal](https://modal.com) ·
🧪 every number traces to [results/results.jsonl](results/results.jsonl) (172 runs)

## Summary
- **Problem.** Mamba-2 decodes with a constant-size state but is weak at exact dense recall; attention
  restores recall but its KV cache grows. Existing Mamba-attention hybrids either use *standard*
  attention or build a Mamba+MLA model by distillation / at production scale, so no one runs a
  controlled single-variable swap of the attention partner.
- **Approach.** Lock depth, width, tokenizer, data order, and total parameters (matched within 0.27%);
  change *only* the attention partner of a shared Mamba-2 backbone. Pretrain on 2.0B FineWeb-Edu tokens;
  probe dense recall with fresh-data MQAR and passkey extrapolation.
- **Key results.**
  - **Cache:** the MLA hybrid decode cache is **5–30x smaller than full attention** (80 MB vs 2416 MB
    at 64k context, size-S) at **statistically equivalent perplexity**; the SWA↔MLA cache crossover is
    located at ~2730 tokens.
  - **Quality:** size-S 3-seed perplexity is tied across the Mamba hybrids (mamba\_mla **71.0 ± 1.8** vs
    mamba\_full 70.3 ± 2.0 vs mamba\_swa 69.8 ± 2.8; pairwise Welch n.s.).
  - **Honest null:** sweeping the MLA latent dim `d_c` over a dense-recall grid shows **no
    minimum-latent knee at this scale**, because MLA caches the latent *per token*, making even
    `d_c=16` non-binding for ≤64 key-value pairs at length 512. We give the mechanism and the regime
    where `d_c` would bind.

## Results (size-S, 125M; full numbers in `results/`)

| variant | FineWeb ppl ↓ | cache @64k (MB) ↓ | passkey @4k ↑ |
|---|---|---|---|
| transformer (full attn) | 96.19 | 2415.9 | 0.06 (n=1) |
| mamba\_full (Jamba)      | 70.34 ± 1.96 | 406.7 | ~0.85 |
| mamba\_swa (Samba)       | **69.79 ± 2.82** | 7.2 (bounded) | ~0.67 |
| **mamba\_mla (ours)**    | 71.00 ± 1.75 | **79.5** | **0.94** |
| mamba2 (pure)            | 74.54 | 4.8 (constant) | 0.57 |

The three Mamba hybrids are statistically tied on quality; mamba\_mla holds the smallest *growing* cache
among the variants that retain global exact recall, and extrapolates passkey beyond its pretraining
context where the pure dense Transformer collapses (single-seed reference).

## Repository layout
- `src/` — training/eval code (`modal_app.py`, one task-dispatched file), config generators
  (`gen_configs.py`, `gen_eval_configs.py`), the run configs (`configs/`), result consolidation
  (`consolidate_results.py`, `aggregate.py`), and plotting (`make_plots.py`).
- `results/` — the append-only `results.jsonl` ledger (172 runs), `results_summary.md`, and the full
  `experiments_log.md` (including the memorization-bug diagnosis and fix).
- `paper/` — LaTeX source, the six figures, and the compiled `paper.pdf`.

## Reproduce
```bash
pip install -r requirements.txt && modal setup
# pretokenize FineWeb-Edu -> Modal Volume, then pretrain a variant:
modal run src/modal_app.py --config-json "@src/configs/pretokenize.json"
modal run src/modal_app.py --config-json "@src/configs/S_mamba_mla_s0.json"
# Track-B synthetic dense-recall sweep (L4 fan-out):
modal run src/modal_app.py --config-json "@src/configs/B_h4_dc_sweep.json"
# consolidate + plot:
python src/consolidate_results.py && python src/aggregate.py && python src/make_plots.py
```
The Mamba-2 CUDA kernel (`mamba-ssm`) and `causal-conv1d` are installed in the Modal image from prebuilt
wheels; a verified pure-PyTorch SSD fallback is used uniformly if the kernel is absent (recorded per run
as `mamba_impl`).

## Citation
```bibtex
@misc{dwivedi2026mambamla,
  title  = {Compressed-Latent vs. Sliding-Window vs. Full Attention as Mamba-2's Partner:
            A Controlled Quality-Cache-Recall Characterization at 125-350M Parameters},
  author = {Dwivedi, Naman and Dandekar, Raj and Dandekar, Rajat and Panat, Sreedath},
  year   = {2026},
  note   = {Vizuara AI Labs}
}
```

## License
MIT (see [LICENSE](LICENSE)).
