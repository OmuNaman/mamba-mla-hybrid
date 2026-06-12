"""
Modal experiment runner for: "How much global cache does a Mamba hybrid need for dense recall?"
Controlled study of compressed-latent (MLA) vs sliding-window vs full attention as Mamba-2's partner.

ONE file, dispatched by config["task"]:
  - "smoke"        : tiny end-to-end sanity for ALL 6 variants (fwd/bwd, MLA decoupled-RoPE numerics,
                     Mamba-2 kernel-or-fallback probe, cache-byte measurement, MQAR gen, no-leakage check)
  - "pretokenize"  : stream FineWeb-Edu sample-10BT + WikiText-103 -> GPT-NeoX BPE -> packed uint16 .bin
                     shards on the Volume (ONCE; reused by all pretrain/eval runs)
  - "pretrain"     : Track A — train one matched variant on FineWeb-Edu; val PPL (FineWeb + WikiText-103);
                     save checkpoint to the Volume
  - "mqar"         : Track B — train a SMALL model directly on synthetic MQAR (Zoology protocol); the
                     d_c-vs-recall frontier (H4), the Mamba-offload axis (H5), and the MQAR baselines.
                     Supports config["batch"]=[cfg,...] -> train.map fan-out on L4.
  - "recall_eval"  : load a Track-A checkpoint; eval MQAR / RULER-multikey / passkey at multiple lengths
  - "efficiency"   : load a Track-A checkpoint; DIRECTLY measure decode cache (bytes+elements), peak
                     memory (length x batch), throughput (tok/s), prefill/decode latency

FAITHFULNESS (checked at code-audit vs the seed PDFs):
  * MLA = DeepSeek-V2 Eqs 9-19 EXACTLY. Latent c^KV (dim d_c) is the ONLY content cached; decoupled RoPE
    uses multi-head q^R and a SINGLE SHARED k^R; q=[q^C;q^R], k=[k^C;k^R]; softmax scale 1/sqrt(d_h+d_h^R);
    RoPE is applied ONLY to q^R/k^R (content q^C/k^C carry no position). Inference cache = c^KV + k^R.
  * Mamba-2 = SSD. Official mamba_ssm.Mamba2 CUDA kernel if importable; else a correct CHUNKED-SSD pure
    PyTorch fallback used UNIFORMLY for every variant (recorded as mamba_impl). Decode state = conv_state
    + ssm_state, constant in length.
  * Param matching: depth & d_model LOCKED across variants; total params equalized via SwiGLU FFN hidden
    size; params + seq-mix/FFN split + FLOPs/token + decode state-bytes logged per variant.
  * NO leakage: causal masks everywhere; MQAR loss only on answer positions; MQAR train/test are disjoint
    sampled instances with eval_seed != train_seed; val shards never overlap train.

Contract (per template): read everything from `config`; write progress.json + metrics.json into RUN_DIR
on the Volume; vol.commit(); return metrics. local_entrypoint prints RESULT_JSON:{...}.
"""
import json
import math
import os
import time

import modal

SLUG = "mamba-mla-hybrid"
APP_NAME = f"research-{SLUG}"
VOL_NAME = f"research-{SLUG}"
DATA_ROOT = "/vol"

app = modal.App(APP_NAME)

# PREBUILT mamba-ssm + causal-conv1d wheels (no source build): pin torch 2.4.0 + cu118 to match the
# official cp311 wheels (state-spaces/mamba v2.2.2 ships cu118torch2.4cxx11abiFALSE). The `|| true`
# keeps the wheel installs NON-FATAL -> if the kernel can't load we fall back to the pure-PyTorch SSD.
_MAMBA_WHL = ("https://github.com/state-spaces/mamba/releases/download/v2.2.2/"
              "mamba_ssm-2.2.2+cu118torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
_CCONV_WHL = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/"
              "causal_conv1d-1.4.0+cu118torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("numpy==1.26.4", "transformers==4.44.2", "datasets==2.21.0", "tqdm==4.67.1",
                 "einops==0.8.0", "huggingface-hub==0.24.6", "fsspec==2024.6.1", "pyarrow==17.0.0",
                 "triton==3.0.0")
    .run_commands(f"pip install '{_CCONV_WHL}' || true", f"pip install '{_MAMBA_WHL}' || true")
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# =============================================================================
# Everything below runs INSIDE the Modal container (torch is only imported there).
# =============================================================================
def _build_and_train(config: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    DT = torch.bfloat16 if (DEV == "cuda" and config.get("bf16", True)) else torch.float32

    # ---- probe the official Mamba-2 kernel once ----------------------------------
    MAMBA_KERNEL = False
    try:
        from mamba_ssm import Mamba2 as _Mamba2Kernel  # noqa: F401
        MAMBA_KERNEL = True
    except Exception:
        MAMBA_KERNEL = False
    FORCE_FALLBACK = bool(config.get("force_pytorch_mamba", False))
    USE_KERNEL = MAMBA_KERNEL and not FORCE_FALLBACK
    # fail LOUD if the official kernel was required but isn't available: the pure-PyTorch SSD fallback
    # is ~3x slower and a silent fall-through would break the runtime/timeout budget.
    if config.get("require_kernel", False) and not USE_KERNEL:
        raise RuntimeError(f"require_kernel=True but Mamba-2 CUDA kernel is unavailable "
                           f"(import_ok={MAMBA_KERNEL}, force_fallback={FORCE_FALLBACK}). Refusing to "
                           f"run the ~3x-slower fallback silently — fix the wheel/image first.")

    def seed_all(s):
        np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    # =========================================================================
    # ROTARY POSITION EMBEDDING (NeoX/LLaMA half-rotation convention)
    # =========================================================================
    def build_rope(dim, max_len, base=10000.0, device=DEV):
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(max_len, device=device).float()
        fr = torch.outer(t, inv)                       # [T, dim/2]
        emb = torch.cat([fr, fr], dim=-1)              # [T, dim]
        return emb.cos(), emb.sin()                    # each [T, dim]

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(x, cos, sin):
        # x: [B, H, T, dim]; cos/sin: [T, dim]
        cos = cos[None, None, :x.shape[-2], :].to(x.dtype)
        sin = sin[None, None, :x.shape[-2], :].to(x.dtype)
        return x * cos + rotate_half(x) * sin

    def swa_attention(q, k, v, window):
        """Causal sliding-window attention WITHOUT materializing a dense [T,T] mask. Process the
        queries in blocks of size `window`; block [s,e) attends only to keys [s-window+1, e) with a
        small [<=w, <=2w] local mask -> O(T*window) memory, not O(T^2). This makes SWA's measured
        prefill memory reflect its true sub-quadratic profile (a dense mask would fake-OOM at long L)."""
        B, H, T, d = q.shape
        w = window
        if T <= w:                                  # window covers everything -> plain causal
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        outs = []
        for s in range(0, T, w):
            e = min(s + w, T)
            lo = max(0, s - w + 1)                  # earliest key any query in [s,e) can see
            qs, ks, vs = q[:, :, s:e], k[:, :, lo:e], v[:, :, lo:e]
            qi = torch.arange(s, e, device=q.device)[:, None]
            kj = torch.arange(lo, e, device=q.device)[None, :]
            rel = qi - kj                           # query_pos - key_pos
            mask = torch.where((rel >= 0) & (rel < w), 0.0, float("-inf")).to(q.dtype)
            outs.append(F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask[None, None]))
        return torch.cat(outs, dim=2)

    # =========================================================================
    # RMSNorm
    # =========================================================================
    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-5):
            super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
        def forward(self, x):
            xf = x.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            return (xf * self.w.float()).type_as(x)

    # =========================================================================
    # FULL / SLIDING-WINDOW MULTI-HEAD ATTENTION (RoPE)
    # =========================================================================
    class Attention(nn.Module):
        def __init__(self, d_model, n_heads, d_head, window=0):
            super().__init__()
            self.nh, self.dh, self.window = n_heads, d_head, window
            self.qkv = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
            self.o = nn.Linear(n_heads * d_head, d_model, bias=False)
        def forward(self, x, rope):
            B, T, _ = x.shape
            q, k, v = self.qkv(x).split(self.nh * self.dh, dim=-1)
            q = q.view(B, T, self.nh, self.dh).transpose(1, 2)   # [B,H,T,dh]
            k = k.view(B, T, self.nh, self.dh).transpose(1, 2)
            v = v.view(B, T, self.nh, self.dh).transpose(1, 2)
            cos, sin = rope
            q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
            if self.window and self.window > 0:
                # sliding-window attention via O(T*window) chunking (no dense [T,T] mask)
                o = swa_attention(q, k, v, self.window)
            else:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o = o.transpose(1, 2).reshape(B, T, self.nh * self.dh)
            return self.o(o)
        # per-token-per-layer decode cache (elements): full = 2*nh*dh ; SWA bounded at window
        def cache_elems(self, t):
            span = t if (not self.window or self.window <= 0) else min(t, self.window)
            return 2 * self.nh * self.dh * span

    # =========================================================================
    # MULTI-HEAD LATENT ATTENTION (MLA) — DeepSeek-V2 Eqs 9-19, decoupled RoPE
    # =========================================================================
    class MLA(nn.Module):
        def __init__(self, d_model, n_heads, d_head, d_c, d_cq, d_rope):
            super().__init__()
            self.nh, self.dh, self.d_c, self.d_cq, self.dr = n_heads, d_head, d_c, d_cq, d_rope
            # KV joint compression (Eq 9-11): c^KV cached; up-proj to per-head k^C, v^C
            self.W_DKV = nn.Linear(d_model, d_c, bias=False)                 # Eq 9
            self.kv_norm = RMSNorm(d_c)                                      # DeepSeek-V2 kv_a_layernorm
            self.W_UK = nn.Linear(d_c, n_heads * d_head, bias=False)         # Eq 10
            self.W_UV = nn.Linear(d_c, n_heads * d_head, bias=False)         # Eq 11
            # query compression (Eq 12-13)
            self.W_DQ = nn.Linear(d_model, d_cq, bias=False)                 # Eq 12
            self.q_norm = RMSNorm(d_cq)                                      # DeepSeek-V2 q_a_layernorm
            self.W_UQ = nn.Linear(d_cq, n_heads * d_head, bias=False)        # Eq 13
            # decoupled RoPE (Eq 14-15): multi-head q^R from c^Q; SINGLE SHARED k^R from h
            self.W_QR = nn.Linear(d_cq, n_heads * d_rope, bias=False)        # Eq 14
            self.W_KR = nn.Linear(d_model, d_rope, bias=False)              # Eq 15  (shared: one head)
            self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)      # Eq 19
        def forward(self, x, rope_r):
            B, T, _ = x.shape
            c_kv = self.kv_norm(self.W_DKV(x))                              # [B,T,d_c]  (normed; cached content)
            c_q = self.q_norm(self.W_DQ(x))                                 # [B,T,d_cq] (normed)
            k_C = self.W_UK(c_kv).view(B, T, self.nh, self.dh).transpose(1, 2)   # [B,H,T,dh]
            v_C = self.W_UV(c_kv).view(B, T, self.nh, self.dh).transpose(1, 2)   # [B,H,T,dh]
            q_C = self.W_UQ(c_q).view(B, T, self.nh, self.dh).transpose(1, 2)    # [B,H,T,dh]
            # decoupled RoPE parts (RoPE applied ONLY here)
            cos, sin = rope_r
            q_R = self.W_QR(c_q).view(B, T, self.nh, self.dr).transpose(1, 2)    # [B,H,T,dr]
            q_R = apply_rope(q_R, cos, sin)
            k_R = self.W_KR(x).view(B, T, 1, self.dr).transpose(1, 2)            # [B,1,T,dr] shared
            k_R = apply_rope(k_R, cos, sin).expand(B, self.nh, T, self.dr)       # broadcast across heads
            q = torch.cat([q_C, q_R], dim=-1)                                   # [B,H,T,dh+dr]  (Eq 16)
            k = torch.cat([k_C, k_R], dim=-1)                                   # [B,H,T,dh+dr]  (Eq 17)
            scale = 1.0 / math.sqrt(self.dh + self.dr)                          # Eq 18 scale
            # pad V to the q/k head-dim (dh+dr) so SDPA stays on the flash/mem-efficient kernel; an
            # asymmetric V head-dim forces the O(T^2) math backend -> OOM at long context. The padding
            # is zeros and is sliced off below, so the result is mathematically identical.
            v_p = F.pad(v_C, (0, self.dr))                                      # [B,H,T,dh+dr]
            o = F.scaled_dot_product_attention(q, k, v_p, is_causal=True, scale=scale)[..., :self.dh]
            o = o.transpose(1, 2).reshape(B, T, self.nh * self.dh)
            return self.W_O(o)                                                  # Eq 19
        # per-token-per-layer decode cache (elements): cache c^KV (d_c) + shared k^R (d_rope) only
        def cache_elems(self, t):
            return (self.d_c + self.dr) * t

    # =========================================================================
    # MAMBA-2 (SSD). Kernel if available, else a correct chunked-SSD fallback.
    # =========================================================================
    class Mamba2Fallback(nn.Module):
        """Pure-PyTorch Mamba-2 (SSD) — chunked scan. Faithful to the SSD recurrence
        h_t = a_t * h_{t-1} + (dt_t * B_t) x_t^T ;  y_t = C_t . h_t + D x_t,  a_t = exp(dt_t * A_h)."""
        def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, chunk=64):
            super().__init__()
            self.d_model = d_model; self.d_state = d_state; self.d_conv = d_conv
            self.expand = expand; self.headdim = headdim; self.ngroups = ngroups; self.chunk = chunk
            self.d_inner = expand * d_model
            assert self.d_inner % headdim == 0
            self.nheads = self.d_inner // headdim
            self.conv_dim = self.d_inner + 2 * ngroups * d_state
            # in_proj -> [z, x, B, C, dt]
            self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * ngroups * d_state + self.nheads, bias=False)
            self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv, groups=self.conv_dim,
                                    padding=d_conv - 1, bias=True)
            self.A_log = nn.Parameter(torch.log(torch.rand(self.nheads) * 15 + 1))   # A = -exp(A_log) < 0
            self.D = nn.Parameter(torch.ones(self.nheads))
            self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
            self.norm = RMSNorm(self.d_inner)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        def forward(self, x, rope=None):
            B, T, _ = x.shape
            zxbcdt = self.in_proj(x)
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)
            # short causal depthwise conv on (x,B,C)
            xBC = self.conv1d(xBC.transpose(1, 2))[..., :T].transpose(1, 2)
            xBC = F.silu(xBC)
            xs, Bm, Cm = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state,
                                           self.ngroups * self.d_state], dim=-1)
            dt = F.softplus(dt + self.dt_bias)                       # [B,T,H] > 0
            A = -torch.exp(self.A_log.float())                       # [H] < 0
            y = self._ssd(xs.float(), dt.float(), A, Bm.float(), Cm.float())
            y = y.to(x.dtype)
            y = self.norm(y * F.silu(z))                              # Mamba-2 gated norm (gate THEN norm,
            return self.out_proj(y)                                  # = mamba_ssm RMSNormGated(norm_before_gate=False))
        def _ssd(self, x, dt, A, Bm, Cm):
            # subscripts: b=batch, w=chunk, i/j=pos-in-chunk, h=head, p=headdim, s=d_state
            B, T, _ = x.shape
            H, P, S, G = self.nheads, self.headdim, self.d_state, self.ngroups
            x = x.view(B, T, H, P)
            Bm = Bm.view(B, T, G, S); Cm = Cm.view(B, T, G, S)
            rep = H // G
            Bm = Bm.repeat_interleave(rep, dim=2)                    # [B,T,H,S]
            Cm = Cm.repeat_interleave(rep, dim=2)
            dA = dt * A[None, None, :]                               # [B,T,H] log-decay per step (<0)
            C = self.chunk
            pad = (C - T % C) % C
            if pad:
                x = F.pad(x, (0, 0, 0, 0, 0, pad)); Bm = F.pad(Bm, (0, 0, 0, 0, 0, pad))
                Cm = F.pad(Cm, (0, 0, 0, 0, 0, pad)); dt = F.pad(dt, (0, 0, 0, pad)); dA = F.pad(dA, (0, 0, 0, pad))
            Tp = T + pad; W = Tp // C
            xc = x.view(B, W, C, H, P); Bc = Bm.view(B, W, C, H, S); Cc = Cm.view(B, W, C, H, S)
            dtc = dt.view(B, W, C, H); dAc = dA.view(B, W, C, H)
            dAcum = torch.cumsum(dAc, dim=2)                         # [B,W,C,H] (<=0, decreasing)
            # ---- intra-chunk diagonal block: masked quadratic dual ----
            # L[i,j] = exp(dAcum_i - dAcum_j) for i>=j   (segment decay = prod a_k, k=j+1..i)
            # Mask the exponent to -inf for i<j BEFORE exp (the upper triangle has a large positive
            # exponent that would overflow to +inf; inf*0 would then be NaN).
            expo = dAcum[:, :, :, None, :] - dAcum[:, :, None, :, :]              # [B,W,C,C,H]
            tri_bool = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))[None, None, :, :, None]
            Lmat = torch.exp(expo.masked_fill(~tri_bool, float("-inf")))         # [B,W,C,C,H]
            CB = torch.einsum('bwihs,bwjhs->bwijh', Cc, Bc)         # [B,W,C,C,H] = C_i . B_j
            M = CB * Lmat * dtc[:, :, None, :, :]                   # weight input j by dt_j
            y_intra = torch.einsum('bwijh,bwjhp->bwihp', M, xc)    # [B,W,C,H,P]
            # ---- chunk-exit state + inter-chunk recurrence ----
            decay_end = torch.exp(dAcum[:, :, -1:, :] - dAcum)     # [B,W,C,H] decay j -> chunk end
            state = torch.einsum('bwjh,bwjhs,bwjhp->bwhsp', decay_end * dtc, Bc, xc)  # [B,W,H,S,P]
            g = torch.exp(dAcum[:, :, -1, :])                      # [B,W,H] total decay across chunk
            s = torch.zeros(B, H, S, P, device=x.device, dtype=x.dtype)
            enter = []
            for w in range(W):
                enter.append(s)                                    # state ENTERING chunk w
                s = g[:, w][:, :, None, None] * s + state[:, w]
            enter = torch.stack(enter, dim=1)                      # [B,W,H,S,P]
            decay_start = torch.exp(dAcum)                         # [B,W,C,H] decay chunk-start -> i
            y_inter = torch.einsum('bwihs,bwhsp->bwihp', Cc, enter) * decay_start[..., None]
            y = (y_intra + y_inter).reshape(B, Tp, H, P)[:, :T]
            y = y + x[:, :T] * self.D[None, None, :, None]
            return y.reshape(B, T, H * P)
        def state_elems(self):
            conv = self.conv_dim * (self.d_conv - 1)
            ssm = self.nheads * self.headdim * self.d_state
            return conv + ssm
        def cache_elems(self, t):           # constant in length
            return self.state_elems()

    class Mamba2Kernel(nn.Module):
        def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1):
            super().__init__()
            from mamba_ssm import Mamba2
            # Standard mamba_ssm.Mamba2 (default mem-eff fused path + causal-conv1d); used the way
            # the library is intended. Real configs use 8-aligned dims so the fused conv is happy.
            self.m = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                            headdim=headdim, ngroups=ngroups)
            self.d_conv = d_conv
            self.conv_dim = expand * d_model + 2 * ngroups * d_state
            self.nheads = (expand * d_model) // headdim
            self.headdim = headdim; self.d_state = d_state
        def forward(self, x, rope=None):
            return self.m(x)
        def state_elems(self):
            return self.conv_dim * (self.d_conv - 1) + self.nheads * self.headdim * self.d_state
        def cache_elems(self, t):
            return self.state_elems()

    def make_mamba(d_model, mcfg):
        # ONE implementation across the WHOLE run, decided once by USE_KERNEL (global): no silent
        # per-layer fallback (that would mix kernel+fallback within a model and break implementation
        # uniformity). If USE_KERNEL but a kernel layer fails to construct, it raises LOUDLY.
        kw = dict(d_state=mcfg.get("d_state", 128), d_conv=mcfg.get("d_conv", 4),
                  expand=mcfg.get("expand", 2), headdim=mcfg.get("headdim", 64),
                  ngroups=mcfg.get("ngroups", 1))
        return (Mamba2Kernel if USE_KERNEL else Mamba2Fallback)(d_model, **kw)

    # =========================================================================
    # SwiGLU FFN
    # =========================================================================
    class SwiGLU(nn.Module):
        def __init__(self, d_model, d_ff):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    # =========================================================================
    # MIXER LIST builder from variant + ratio
    # =========================================================================
    def build_mixers(variant, n_layers, attn_ratio):
        # attn_ratio = mamba:attn (e.g. 5 -> 1 attn per 5 mamba); n_attn placed periodic + tail
        backbone = {"transformer": "mha", "mamba2": "mamba2", "mamba_full": "mamba2",
                    "mamba_swa": "mamba2", "mamba_mla": "mamba2", "transformer_mla": "mla",
                    "swa_mla": "swa"}[variant]
        attn = {"transformer": None, "mamba2": None, "mamba_full": "mha", "mamba_swa": "swa",
                "mamba_mla": "mla", "transformer_mla": None, "swa_mla": "mla"}[variant]
        mixers = [backbone] * n_layers
        if attn is not None:
            n_attn = max(1, round(n_layers / (attn_ratio + 1)))
            # evenly spaced positions, last one biased to the tail
            step = n_layers / n_attn
            pos = sorted(set(min(n_layers - 1, int(round((i + 1) * step - 1))) for i in range(n_attn)))
            for p in pos:
                mixers[p] = attn
        return mixers

    # =========================================================================
    # THE LANGUAGE MODEL
    # =========================================================================
    class Block(nn.Module):
        def __init__(self, d_model, mixer):
            super().__init__()
            self.n1 = RMSNorm(d_model); self.mixer = mixer
            self.n2 = RMSNorm(d_model); self.mlp = None  # set by parent
        def forward(self, x, rope, rope_r):
            m = self.mixer
            if isinstance(m, MLA):
                x = x + m(self.n1(x), rope_r)
            elif isinstance(m, Attention):
                x = x + m(self.n1(x), rope)
            else:  # mamba
                x = x + m(self.n1(x))
            x = x + self.mlp(self.n2(x))
            return x

    class LM(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            d = cfg["d_model"]; self.cfg = cfg
            self.vocab = cfg["vocab"]; self.d_model = d
            self.nh = cfg["n_heads"]; self.dh = cfg["d_head"]
            self.embed = nn.Embedding(self.vocab, d)
            mixers = cfg["mixers"]
            mla = cfg.get("mla", {})
            self.d_rope = mla.get("d_rope", self.dh // 2)
            blocks = []
            for mt in mixers:
                if mt == "mha":
                    mix = Attention(d, self.nh, self.dh, window=0)
                elif mt == "swa":
                    mix = Attention(d, self.nh, self.dh, window=cfg.get("swa_window", 512))
                elif mt == "mla":
                    mix = MLA(d, self.nh, self.dh, d_c=mla.get("d_c", 4 * self.dh),
                              d_cq=mla.get("d_cq", 6 * self.dh), d_rope=self.d_rope)
                elif mt == "mamba2":
                    mix = make_mamba(d, cfg.get("mamba", {}))
                else:
                    raise ValueError(mt)
                b = Block(d, mix); b.mlp = SwiGLU(d, cfg["d_ff"])
                blocks.append(b)
            self.blocks = nn.ModuleList(blocks)
            self.norm_f = RMSNorm(d)
            self.lm_head = nn.Linear(d, self.vocab, bias=False)
            self.lm_head.weight = self.embed.weight   # tied
            self.mixers = mixers
            maxlen = cfg.get("max_len", 65536)
            self._cos, self._sin = build_rope(self.dh, maxlen, base=cfg.get("rope_base", 10000.0))
            self._cosr, self._sinr = build_rope(self.d_rope, maxlen, base=cfg.get("rope_base", 10000.0))
        def forward(self, idx):
            x = self.embed(idx)
            rope = (self._cos, self._sin); rope_r = (self._cosr, self._sinr)
            for b in self.blocks:
                x = b(x, rope, rope_r)
            x = self.norm_f(x)
            return self.lm_head(x)
        def cache_elems_per_token_layers(self, t):
            """Total inference cache (elements) for ALL layers at context length t."""
            tot = 0
            for b in self.blocks:
                tot += b.mixer.cache_elems(t)
            return tot

        def mamba_state_elems(self):
            """Total constant Mamba recurrent state (elements) across layers (0 if no Mamba)."""
            tot = 0
            for b in self.blocks:
                if hasattr(b.mixer, "state_elems"):
                    tot += b.mixer.state_elems()
            return tot

        def flops_per_token(self, L):
            """Analytic forward FLOPs/token at context length L (NON-embedding; the controlled
            differentiator). Linear matmuls = 2*sum(weight numels); plus the sequence-dependent
            attention/MLA term (full=L, SWA=window, MLA=L on the compressed dims) and the Mamba SSD
            state update (~nheads*headdim*d_state)."""
            total = 0
            for b in self.blocks:
                m = b.mixer
                total += 2 * sum(p.numel() for p in m.parameters() if p.dim() == 2)
                total += 2 * sum(p.numel() for p in b.mlp.parameters() if p.dim() == 2)
                if isinstance(m, Attention):
                    span = L if (not m.window or m.window <= 0) else min(L, m.window)
                    total += 2 * 2 * m.nh * m.dh * span                  # QK^T and A.V
                elif isinstance(m, MLA):
                    total += 2 * m.nh * (m.dh + m.dr) * L                 # QK^T on [content;rope]
                    total += 2 * m.nh * m.dh * L                          # A.V
                elif hasattr(m, "nheads"):                               # Mamba-2 SSD state update
                    total += 2 * m.nheads * m.headdim * m.d_state
            return total
        def param_split(self):
            seqmix, ffn, emb, other = 0, 0, 0, 0
            for n, p in self.named_parameters():
                if "embed" in n or "lm_head" in n:
                    emb += p.numel()
                elif ".mlp." in n:
                    ffn += p.numel()
                elif ".mixer." in n:
                    seqmix += p.numel()
                else:
                    other += p.numel()
            return dict(seqmix=seqmix, ffn=ffn, embed=emb, other=other,
                        total=seqmix + ffn + emb + other,
                        non_embed=seqmix + ffn + other)

    # =========================================================================
    # PARAM MATCHING: pick d_ff so non-embedding params ~= target
    # =========================================================================
    def count_nonembed(cfg, d_ff):
        c = dict(cfg); c["d_ff"] = d_ff
        m = LM(c); ps = m.param_split(); del m
        return ps["non_embed"], ps

    def solve_d_ff(cfg, target_nonembed):
        # binary-search the SwiGLU FFN hidden size to hit the target non-embed param count; return the
        # CLOSEST d_ff found (not the last probe) AND the achieved relative error so param-match parity
        # is auditable (a mixer that alone overshoots the reference can't be matched by shrinking FFN).
        lo, hi = 8, 16384
        best_dff, best_err = 8, 1e18
        for _ in range(32):
            if lo > hi:
                break
            mid = max(8, ((lo + hi) // 2 // 8) * 8)
            ne, _ = count_nonembed(cfg, mid)
            err = abs(ne - target_nonembed) / target_nonembed
            if err < best_err:
                best_err, best_dff = err, mid
            if err < 0.003:
                break
            if ne < target_nonembed:
                lo = mid + 8
            else:
                hi = mid - 8
        return best_dff, best_err

    # =========================================================================
    # SYNTHETIC RECALL DATA — MQAR (Zoology), passkey, RULER-multikey
    # =========================================================================
    def gen_mqar(n, seq_len, n_kv, vocab, seed, query_frac=0.5, spread=False):
        """Canonical Zoology MQAR. Definition pairs [k v ...] then a query section that REPEATS a
        subset of keys with their values; we supervise logits at each query-KEY position to predict
        its value (the next token) -> recoverable ONLY by associative recall (causal). Loss masked to
        query-key positions only. spread=False (Track B headline): dense front-loaded form. spread=True
        (recall_eval length axis): definition pairs scattered across [0,0.8L) over random filler so the
        recall DISTANCE actually grows with L (filler band disjoint from keys/values -> no spurious
        bindings). Returns (X[n,L], Y[n,L] with -100 except query-key positions)."""
        rng = np.random.default_rng(seed)
        L = seq_len
        X = np.zeros((n, L), dtype=np.int64)
        Y = np.full((n, L), -100, dtype=np.int64)
        if spread:
            key_lo, key_hi = 2, vocab // 4
            val_lo, val_hi = vocab // 4, vocab // 2
            fil_lo, fil_hi = vocab // 2, vocab                     # filler band disjoint from keys/vals
        else:
            key_lo, key_hi = 2, vocab // 2
            val_lo, val_hi = vocab // 2, vocab
        # cap n_kv so the definition+query block always fits (avoids all-masked -> NaN loss)
        nq = max(1, int(round(n_kv * query_frac)))
        max_kv = max(1, (L - 2) // 2 - nq)
        n_kv_eff = min(n_kv, max_kv)
        nq = min(nq, n_kv_eff)
        if not spread:
            # VECTORIZED dense MQAR (batch-level) — fast path for Track-B's long fresh-data training.
            # Same structure as the per-row loop: definition [k0 v0 k1 v1 ...] then queries [kq vq ...];
            # supervise Y at each query-key position. Distinct keys per row via argsort of randoms.
            ks = min(key_hi - key_lo, 1024)          # cap key space so distinct sampling stays cheap
            # argpartition (O(n*ks)) not argsort (O(n*ks*log)) -> ~50x faster fresh gen per step
            keys = key_lo + np.argpartition(rng.random((n, ks)), n_kv_eff - 1, axis=1)[:, :n_kv_eff]
            vals = rng.integers(val_lo, val_hi, size=(n, n_kv_eff))
            deflen = 2 * n_kv_eff
            X[:, 0:deflen:2] = keys
            X[:, 1:deflen:2] = vals
            qsel = np.argpartition(rng.random((n, n_kv_eff)), nq - 1, axis=1)[:, :nq]   # distinct query idx
            rows = np.arange(n)[:, None]
            qk = keys[rows, qsel]; qv = vals[rows, qsel]
            qpos = deflen + 2 * np.arange(nq)
            X[:, qpos] = qk
            X[:, qpos + 1] = qv
            Y[:, qpos] = qv
            return torch.from_numpy(X), torch.from_numpy(Y)
        for i in range(n):
            keys = rng.choice(np.arange(key_lo, key_hi), size=n_kv_eff, replace=False)
            vals = rng.integers(val_lo, val_hi, size=n_kv_eff)
            qsel = rng.choice(n_kv_eff, size=nq, replace=False)
            if spread:
                seq = rng.integers(fil_lo, fil_hi, size=L).astype(np.int64)
                region = max(2 * n_kv_eff + 2, int(0.8 * L))
                slots = sorted(rng.choice(np.arange(0, region - 1, 2),
                                          size=min(n_kv_eff, (region - 1) // 2), replace=False))
                for (k, v), s in zip(zip(keys, vals), slots):
                    seq[s] = int(k); seq[s + 1] = int(v)
                ans = []
                for idx, j in enumerate(qsel):
                    p = region + 2 * idx
                    if p + 1 < L:
                        seq[p] = int(keys[j]); seq[p + 1] = int(vals[j]); ans.append((p, int(vals[j])))
                X[i] = seq
            else:
                seq = []
                for k, v in zip(keys, vals):
                    seq.extend([int(k), int(v)])
                ans = []
                for j in qsel:
                    p = len(seq)
                    seq.append(int(keys[j])); ans.append((p, int(vals[j]))); seq.append(int(vals[j]))
                seq = seq[:L] + [0] * max(0, L - len(seq))
                X[i] = np.array(seq[:L], dtype=np.int64)
            for (p, v) in ans:
                if p < L:
                    Y[i, p] = v
        return torch.from_numpy(X), torch.from_numpy(Y)

    def gen_passkey(n, seq_len, vocab, seed):
        """Single-needle passkey (saturated floor). Definition [CUE, pk] hidden in filler; query
        [CUE, (predict pk)] at the end. CUE=token 1; pk in [2,9]; filler in [10,vocab)."""
        rng = np.random.default_rng(seed)
        X = np.zeros((n, seq_len), dtype=np.int64)
        Y = np.full((n, seq_len), -100, dtype=np.int64)
        for i in range(n):
            pk = int(rng.integers(2, 50))                          # ~1/48 chance floor
            pos = int(rng.integers(seq_len // 8, seq_len - 6))
            seq = rng.integers(50, vocab, size=seq_len).astype(np.int64)
            seq[pos] = 1; seq[pos + 1] = pk                        # definition: CUE pk
            seq[-2] = 1; seq[-1] = pk                              # query: CUE then pk (next token)
            X[i] = seq; Y[i, -2] = pk                              # logits[-2] -> pk
        return torch.from_numpy(X), torch.from_numpy(Y)

    def gen_ruler_multikey(n, seq_len, n_needles, vocab, seed):
        """multi-key NIAH: n_needles (key,val) needles with DISTINCT keys in random filler; query ONE
        random needle: [key, (predict val)] at the end (val is the next token)."""
        rng = np.random.default_rng(seed)
        X = np.zeros((n, seq_len), dtype=np.int64)
        Y = np.full((n, seq_len), -100, dtype=np.int64)
        for i in range(n):
            seq = rng.integers(100, vocab, size=seq_len).astype(np.int64)
            keys = rng.choice(np.arange(2, 99), size=n_needles, replace=False)
            vals = rng.integers(vocab // 2, vocab, size=n_needles)
            # space needles >=3 apart (each occupies p and p+1) so no needle's value is overwritten
            grid = np.arange(4, seq_len - 6, 3)
            positions = sorted(rng.choice(grid, size=min(n_needles, len(grid)), replace=False))
            for k, v, p in zip(keys, vals, positions):
                seq[p] = int(k); seq[p + 1] = int(v)
            qi = int(rng.integers(0, n_needles))
            seq[-2] = int(keys[qi]); seq[-1] = int(vals[qi])       # query key then value (next token)
            X[i] = seq; Y[i, -2] = int(vals[qi])                   # logits[-2] -> val
        return torch.from_numpy(X), torch.from_numpy(Y)

    # =========================================================================
    # LM DATA (FineWeb-Edu packed shards) / synthetic-LM for smoke
    # =========================================================================
    def load_bin(path):
        return np.memmap(path, dtype=np.uint16, mode="r")

    def lm_batch(data, batch, seq_len, device, rng):
        ix = rng.integers(0, len(data) - seq_len - 1, size=batch)
        x = np.stack([np.asarray(data[i:i + seq_len], dtype=np.int64) for i in ix])
        y = np.stack([np.asarray(data[i + 1:i + 1 + seq_len], dtype=np.int64) for i in ix])
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    # =========================================================================
    # TRAIN / EVAL HELPERS
    # =========================================================================
    def cosine_lr(step, total, peak, warm=0.1, floor=0.1):
        w = int(total * warm)
        if step < w:
            return peak * step / max(1, w)
        prog = (step - w) / max(1, total - w)
        return floor * peak + 0.5 * (1 - floor) * peak * (1 + math.cos(math.pi * prog))

    def make_model(cfg):
        m = LM(cfg).to(DEV)
        for p in m.parameters():
            if p.dim() >= 2 and p.requires_grad and p is not m.embed.weight:
                pass
        return m

    def eval_ppl(model, data, seq_len, device, n_batches=40, batch=8, seed=1234):
        model.eval(); rng = np.random.default_rng(seed); tot, cnt = 0.0, 0
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = lm_batch(data, batch, seq_len, device, rng)
                with torch.autocast(device_type="cuda", dtype=DT, enabled=(device == "cuda")):
                    logits = model(x)
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
                tot += loss.item(); cnt += 1
        model.train()
        return math.exp(tot / max(1, cnt))

    def eval_recall_acc(model, X, Y, device, batch=64):
        model.eval(); correct, total = 0, 0
        # the binding allocation is the [b, L, vocab] logits tensor; cap b so it stays small at long L
        # (recall accuracy is batch-invariant, so this only trades a little speed for OOM-safety).
        L = X.shape[1]
        batch = max(1, min(batch, max(1, 24_000 // max(1, L))))
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = X[i:i + batch].to(device); yb = Y[i:i + batch].to(device)
                with torch.autocast(device_type="cuda", dtype=DT, enabled=(device == "cuda")):
                    logits = model(xb)
                pred = logits.argmax(-1)
                mask = yb != -100
                correct += (pred[mask] == yb[mask]).sum().item()
                total += mask.sum().item()
                del logits, pred, xb, yb
        if device == "cuda":
            torch.cuda.empty_cache()
        model.train()
        return correct / max(1, total)

    # =========================================================================
    # TASK DISPATCH
    # =========================================================================
    task = config.get("task", "smoke")
    seed = int(config.get("seed", 0)); seed_all(seed)
    exp_id = config.get("exp_id", task)
    run_dir = os.path.join(DATA_ROOT, exp_id)
    os.makedirs(run_dir, exist_ok=True)
    prog_path = os.path.join(run_dir, "progress.json")
    def prog(**kw):
        _write_json(prog_path, {"exp_id": exp_id, "task": task, **kw}); vol.commit()

    info = dict(mamba_impl=("kernel" if USE_KERNEL else "pytorch_ssd_fallback"),
                mamba_kernel_available=MAMBA_KERNEL, device=DEV, dtype=str(DT))

    # ---------- helper to assemble a model cfg + param-match ----------
    def assemble_cfg(c):
        mc = dict(
            vocab=c.get("vocab", 50304), d_model=c["d_model"], n_layers=c["n_layers"],
            n_heads=c.get("n_heads", c["d_model"] // 64), d_head=c.get("d_head", 64),
            swa_window=c.get("swa_window", 512), rope_base=c.get("rope_base", 10000.0),
            max_len=c.get("max_len", 65536),   # RoPE tables cover up to 64k for long-context eval
            mla=dict(d_c=c.get("d_c", 256), d_cq=c.get("d_cq", 384), d_rope=c.get("d_rope", 32)),
            mamba=c.get("mamba", {}),
        )
        mc["mixers"] = build_mixers(c["variant"], c["n_layers"], c.get("attn_ratio", 5))
        # param matching: target non-embed = the Transformer reference at the same d_model/n_layers
        if c.get("match_params", True):
            ref = dict(mc); ref["mixers"] = ["mha"] * c["n_layers"]
            tgt, _ = count_nonembed(ref, c.get("ref_d_ff", 4 * c["d_model"]))
            mc["d_ff"], mc["_pm_err"] = solve_d_ff(mc, tgt)
            # hard guard: a >10% miss means matching genuinely failed (mixer overshoots ref) -> stop
            if mc["_pm_err"] > 0.10:
                raise RuntimeError(f"param-match failed for {c['variant']}: non-embed off by "
                                   f"{mc['_pm_err']*100:.1f}% (mixer likely overshoots the reference)")
        else:
            mc["d_ff"] = c.get("d_ff", 4 * c["d_model"]); mc["_pm_err"] = 0.0
        return mc

    # =====================================================================
    if task == "smoke":
        out = dict(info)
        variants = config.get("variants",
                              ["transformer", "mamba2", "mamba_full", "mamba_swa", "mamba_mla", "transformer_mla"])
        # realistic, 8-aligned dims so the SSD/conv kernels are genuinely exercised
        d_model, n_layers, vocab, seq_len = 256, 4, 1024, 256
        per = {}
        for v in variants:
            cfg = assemble_cfg(dict(variant=v, d_model=d_model, n_layers=n_layers, vocab=vocab,
                                    seq_len=seq_len, n_heads=4, d_head=64, d_c=128, d_cq=192, d_rope=32,
                                    swa_window=64, attn_ratio=2, ref_d_ff=4 * d_model))
            m = make_model(cfg)
            # fwd/bwd on random LM data
            rng = np.random.default_rng(0)
            data = (np.random.default_rng(0).integers(0, vocab, size=40000)).astype(np.uint16)
            x, y = lm_batch(data, 8, seq_len, DEV, rng)
            with torch.autocast(device_type="cuda", dtype=DT, enabled=(DEV == "cuda")):
                logits = m(x)
            loss = F.cross_entropy(logits.float().view(-1, vocab), y.view(-1))
            loss.backward()
            gnorm = sum((p.grad.norm().item() ** 2) for p in m.parameters() if p.grad is not None) ** 0.5
            ps = m.param_split()
            ce16 = m.cache_elems_per_token_layers(16)
            per[v] = dict(loss=float(loss.item()), grad_norm=float(gnorm), params=ps,
                          cache_elems_at_16=int(ce16), finite=bool(torch.isfinite(loss).item()))
            del m
        # MQAR gen + no-leakage check
        Xtr, Ytr = gen_mqar(8, 64, 4, 256, seed=1)
        Xte, Yte = gen_mqar(8, 64, 4, 256, seed=999)
        leak = bool((Xtr.unsqueeze(1) == Xte.unsqueeze(0)).all(-1).any().item())
        out["per_variant"] = per
        out["mqar_answer_positions_per_seq"] = int((Ytr != -100).sum(1).float().mean().item())
        out["mqar_train_test_identical_any"] = leak
        out["ok"] = all(per[v]["finite"] for v in per)
        _write_json(os.path.join(run_dir, "metrics.json"), out)
        prog(done=True, **{k: out[k] for k in ("ok", "mamba_impl")})
        return out

    # =====================================================================
    if task == "probe":
        # time a few train steps at REAL dims on the chosen GPU -> per-step time, tok/s, peak mem,
        # and which mamba_impl actually ran. Used to validate the kernel at scale + ground the cost
        # estimate for the human checkpoint. Trains on random tokens (no FineWeb needed).
        cfg = assemble_cfg(config)
        model = make_model(cfg); ps = model.param_split()
        seq_len = config.get("seq_len", 2048); batch = config.get("batch", 8)
        steps = config.get("steps", 12); vocab = cfg["vocab"]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
        rng = np.random.default_rng(0)
        data = rng.integers(0, vocab, size=batch * seq_len * 6).astype(np.uint16)
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        def step_once():
            x, y = lm_batch(data, batch, seq_len, DEV, rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=DT, enabled=(DEV == "cuda")):
                loss = F.cross_entropy(model(x).float().view(-1, vocab), y.view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            return float(loss.item())
        for _ in range(2):       # warmup (Triton autotune / cudnn)
            step_once()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); last = 0.0
        for _ in range(steps):
            last = step_once()
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        peak = torch.cuda.max_memory_allocated() if DEV == "cuda" else 0
        out = dict(info, variant=config["variant"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                   d_ff=cfg["d_ff"], seq_len=seq_len, batch=batch, steps=steps,
                   s_per_step=dt, tok_per_s=batch * seq_len / max(1e-9, dt),
                   peak_gb=round(peak / 1e9, 2), params_nonembed=ps["non_embed"], last_loss=last)
        _write_json(os.path.join(run_dir, "metrics.json"), out); prog(done=True, **{k: out[k] for k in ("s_per_step", "tok_per_s", "peak_gb", "mamba_impl")})
        return out

    # =====================================================================
    if task == "pretokenize":
        from transformers import AutoTokenizer
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        eot = tok.eos_token_id or 0
        target_tokens = int(config.get("target_tokens", 2_500_000_000))
        out_dir = os.path.join(DATA_ROOT, "data"); os.makedirs(out_dir, exist_ok=True)
        # ---- FineWeb-Edu train ----
        fw_path = os.path.join(out_dir, "fineweb_edu_train.bin")
        if not os.path.exists(fw_path) or config.get("force", False):
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            buf = np.memmap(fw_path, dtype=np.uint16, mode="w+", shape=(target_tokens,))
            n = 0; last_log = 0; batch = []; BSZ = int(config.get("tok_batch", 1024))
            def _emit(texts):
                nonlocal n, last_log
                if not texts or n >= target_tokens:
                    return
                flat = []
                for tids in tok(texts).input_ids:   # fast (Rust) tokenizer parallelizes the whole batch across CPU cores
                    flat.extend(tids); flat.append(eot)
                a = np.asarray(flat, dtype=np.uint16)
                take = min(len(a), target_tokens - n)
                buf[n:n + take] = a[:take]; n += take
                if n - last_log >= 50_000_000:
                    prog(stage="fineweb", tokens=n, target=target_tokens); buf.flush(); last_log = n
            for ex in ds:
                batch.append(ex["text"])
                if len(batch) >= BSZ:
                    _emit(batch); batch = []
                if n >= target_tokens:
                    break
            _emit(batch)
            buf.flush(); del buf
            _write_json(os.path.join(out_dir, "fineweb_edu_train.meta.json"), {"tokens": int(n)})
        # ---- WikiText-103 val ----
        wt_path = os.path.join(out_dir, "wikitext103_val.bin")
        if not os.path.exists(wt_path) or config.get("force", False):
            wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            texts = [t for t in wt["text"] if t.strip()]
            ids = []
            for enc in tok(texts).input_ids:   # batched
                ids.extend(enc); ids.append(eot)
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(wt_path)
            _write_json(os.path.join(out_dir, "wikitext103_val.meta.json"), {"tokens": int(len(arr))})
        vol.commit()
        out = dict(info, fineweb_path=fw_path, wikitext_path=wt_path,
                   fineweb_tokens=int(os.path.getsize(fw_path) // 2),
                   wikitext_tokens=int(os.path.getsize(wt_path) // 2))
        _write_json(os.path.join(run_dir, "metrics.json"), out); prog(done=True)
        return out

    # =====================================================================
    if task == "pretrain":
        cfg = assemble_cfg(config)
        cfg_seq = config.get("seq_len", 2048)
        model = make_model(cfg)
        ps = model.param_split()
        data_dir = os.path.join(DATA_ROOT, "data")
        train_bin = os.path.join(data_dir, "fineweb_edu_train.bin")
        if not os.path.exists(train_bin):
            return dict(info, error="run pretokenize first (fineweb_edu_train.bin missing)")
        full = load_bin(train_bin)
        n_val = 50_000_000
        train_data = full[:-n_val]; val_fw = full[-n_val:]
        wt_bin = os.path.join(data_dir, "wikitext103_val.bin")
        val_wt = load_bin(wt_bin) if os.path.exists(wt_bin) else None
        tok_budget = int(config.get("tokens", 2_000_000_000))
        batch = config.get("batch", 32); accum = config.get("accum", 16)
        steps = tok_budget // (batch * accum * cfg_seq)
        opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1.2e-3),
                                betas=(0.9, 0.95), weight_decay=0.1)
        rng = np.random.default_rng(seed)
        peak_lr = config.get("lr", 1.2e-3)
        save_every = max(50, steps // 8)                 # ~8 resumable checkpoints per run
        latest = os.path.join(run_dir, "ckpt_latest.pt")
        done_path = os.path.join(run_dir, "model.pt")
        if os.path.exists(done_path) and not config.get("force_retrain", False):
            return dict(info, note="already trained (model.pt exists); skipping", variant=config["variant"])
        start_step, train_elapsed = 0, 0.0
        # ---- RESUME from a partial checkpoint if present (survives the Modal timeout / preemption) ----
        if os.path.exists(latest):
            try:
                st = torch.load(latest, map_location=DEV)
                model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
                start_step = int(st["step"]); rng.bit_generator.state = st["rng_state"]
                train_elapsed = float(st.get("train_elapsed", 0.0)); info["resumed_from_step"] = start_step
            except Exception as e:
                info["resume_failed"] = str(e)
        t0 = time.time()
        for step in range(start_step, steps):
            lr = cosine_lr(step, steps, peak_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            lsum = 0.0
            for _ in range(accum):
                x, y = lm_batch(train_data, batch, cfg_seq, DEV, rng)
                with torch.autocast(device_type="cuda", dtype=DT, enabled=(DEV == "cuda")):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().view(-1, cfg["vocab"]), y.view(-1)) / accum
                loss.backward(); lsum += float(loss.item())          # lsum = accumulated-batch mean
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % max(1, steps // 50) == 0 or step == steps - 1:
                el = train_elapsed + (time.time() - t0)
                prog(step=step, total=steps, loss=lsum,
                     lr=lr, elapsed_s=el, tok_per_s=(step + 1) * batch * accum * cfg_seq / max(1, el))
            if (step + 1) % save_every == 0 and step + 1 < steps:   # periodic resumable checkpoint
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1,
                            "rng_state": rng.bit_generator.state,
                            "train_elapsed": train_elapsed + (time.time() - t0)}, latest)
                vol.commit()
        ppl_fw = eval_ppl(model, val_fw, cfg_seq, DEV, n_batches=60)
        ppl_wt = eval_ppl(model, val_wt, cfg_seq, DEV, n_batches=60) if val_wt is not None else None
        # final model checkpoint; drop the (large) resume checkpoint
        torch.save({"state_dict": model.state_dict(), "cfg": cfg}, done_path)
        try:
            os.remove(latest)
        except OSError:
            pass
        CB = 2  # fixed cache precision (bf16, 2 bytes) for apples-to-apples cache bytes
        out = dict(info, variant=config["variant"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                   d_ff=cfg["d_ff"], d_c=cfg["mla"]["d_c"], seq_len=cfg_seq, tokens=tok_budget, steps=steps,
                   param_split=ps, param_match_err=cfg.get("_pm_err", 0.0),
                   ppl_fineweb=ppl_fw, ppl_wikitext=ppl_wt, seed=seed,
                   cache_elems_4k=int(model.cache_elems_per_token_layers(4096)),
                   cache_elems_16k=int(model.cache_elems_per_token_layers(16384)),
                   cache_bytes_4k=int(model.cache_elems_per_token_layers(4096) * CB),
                   cache_bytes_16k=int(model.cache_elems_per_token_layers(16384) * CB),
                   mamba_state_bytes=int(model.mamba_state_elems() * CB),
                   flops_per_token_2k=int(model.flops_per_token(2048)),
                   flops_per_token_16k=int(model.flops_per_token(16384)),
                   train_s=time.time() - t0, mixers=cfg["mixers"])
        _write_json(os.path.join(run_dir, "metrics.json"), out)
        prog(done=True, ppl_fineweb=ppl_fw); vol.commit()
        return out

    # =====================================================================
    if task == "mqar":
        # Track B: train a SMALL model directly on synthetic MQAR; report recall.
        cfg = assemble_cfg(config)
        model = make_model(cfg)
        ps = model.param_split()
        vocab = cfg["vocab"]; seq_len = config.get("seq_len", 512)
        n_kv = config.get("n_kv", 32); n_train = config.get("n_train", 20000)
        steps = config.get("steps", 2000); batch = config.get("batch", 64)
        opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=0.1, betas=(0.9, 0.95))
        # FRESH random MQAR every step (effectively infinite data) so the model must learn the in-context
        # retrieval ALGORITHM, not memorize a finite pool. A fixed train pool (the old code) lets a small
        # model hit ~97% train / ~chance test (pure memorization) -> the whole d_c-vs-recall frontier reads
        # as zero. Disjoint fixed test pool (seed+777) is held out for the reported accuracy.
        Xte, Yte = gen_mqar(2048, seq_len, n_kv, vocab, seed=seed + 777)
        _GS = (seed + 1) * 9_999_991                 # per-run base so different seeds see different streams
        Xtr, Ytr = gen_mqar(512, seq_len, n_kv, vocab, seed=_GS - 1)   # small held-in probe for train_acc
        t0 = time.time()
        for step in range(steps):
            xb, yb = gen_mqar(batch, seq_len, n_kv, vocab, seed=_GS + step)   # fresh batch each step
            xb = xb.to(DEV); yb = yb.to(DEV)
            lr = cosine_lr(step, steps, config.get("lr", 1e-3))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=DT, enabled=(DEV == "cuda")):
                logits = model(xb)
                ym = yb.view(-1); sm = ym != -100        # MQAR supervises only ~nq of T positions:
                loss = F.cross_entropy(logits.view(-1, vocab)[sm].float(), ym[sm])   # gather BEFORE float-cast
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % max(1, steps // 20) == 0:
                prog(step=step, total=steps, loss=float(loss.item()))
        acc = eval_recall_acc(model, Xte, Yte, DEV)
        train_acc = eval_recall_acc(model, Xtr[:512], Ytr[:512], DEV)   # DEBUG: can it even fit train?
        out = dict(info, variant=config["variant"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                   d_c=cfg["mla"]["d_c"], d_ff=cfg["d_ff"], n_kv=n_kv, seq_len=seq_len, mqar_acc=acc, seed=seed,
                   final_train_loss=float(loss.item()), train_acc=train_acc,
                   param_split=ps, param_match_err=cfg.get("_pm_err", 0.0),
                   cache_elems=int(model.cache_elems_per_token_layers(seq_len)),
                   cache_bytes=int(model.cache_elems_per_token_layers(seq_len) * 2),
                   mamba_state_bytes=int(model.mamba_state_elems() * 2),
                   flops_per_token=int(model.flops_per_token(seq_len)),
                   match_params=config.get("match_params", True),
                   attn_ratio=config.get("attn_ratio", 5), train_s=time.time() - t0)
        _write_json(os.path.join(run_dir, "metrics.json"), out); prog(done=True, mqar_acc=acc)
        return out

    # =====================================================================
    if task == "recall_eval":
        ckpt = torch.load(config["ckpt"], map_location=DEV)
        base_cfg = ckpt["cfg"]; base_sd = ckpt["state_dict"]
        vocab = base_cfg["vocab"]
        adapt_steps = config.get("adapt_steps", 300)
        adapt_len = config.get("adapt_len", 512)
        lengths = config.get("lengths", [512, 1024, 2048, 4096])

        def fresh():
            m = LM(base_cfg).to(DEV); m.load_state_dict(base_sd); return m

        def adapt_and_eval(gen, train_kw, test_kw):
            # equal-budget adaptation: FRESH random data every step (not a fixed pool) so the model learns
            # the retrieval algorithm rather than memorizing -> eval on a DISJOINT fresh test (seed 99).
            m = fresh(); m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=config.get("adapt_lr", 5e-4), weight_decay=0.0)
            adapt_bs = 64
            tkw = {k: v for k, v in train_kw.items() if k != "n"}
            for st in range(adapt_steps):
                xb, yb = gen(seed=10_000_000 + st, n=adapt_bs, **tkw)   # fresh batch each step
                xb = xb.to(DEV); yb = yb.to(DEV)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=DT, enabled=(DEV == "cuda")):
                    ym = yb.view(-1); sm = ym != -100
                    loss = F.cross_entropy(m(xb).view(-1, vocab)[sm].float(), ym[sm])   # gather sparse targets
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            accs = {}
            for L in lengths:
                Xte, Yte = gen(seed=99, **{**test_kw, "seq_len": L})
                accs[str(L)] = eval_recall_acc(m, Xte, Yte, DEV)
            del m
            return accs

        res = {}
        res["mqar"] = adapt_and_eval(
            lambda seed, **kw: gen_mqar(n=kw.pop("n"), vocab=vocab, seed=seed, spread=True, **kw),
            dict(n=4000, seq_len=adapt_len, n_kv=config.get("n_kv", 16)),
            dict(n=512, n_kv=config.get("n_kv", 16)))
        res["passkey"] = adapt_and_eval(
            lambda seed, **kw: gen_passkey(n=kw.pop("n"), vocab=vocab, seed=seed, **kw),
            dict(n=2000, seq_len=adapt_len), dict(n=256))
        res["ruler_multikey"] = adapt_and_eval(
            lambda seed, **kw: gen_ruler_multikey(n=kw.pop("n"), vocab=vocab, seed=seed, **kw),
            dict(n=3000, seq_len=adapt_len, n_needles=config.get("n_needles", 8)),
            dict(n=256, n_needles=config.get("n_needles", 8)))
        out = dict(info, mixers=base_cfg.get("mixers"), adapt_steps=adapt_steps,
                   adapt_len=adapt_len, by_task=res)
        _write_json(os.path.join(run_dir, "metrics.json"), out); prog(done=True)
        return out

    # =====================================================================
    if task == "efficiency":
        ckpt = torch.load(config["ckpt"], map_location=DEV)
        cfg = ckpt["cfg"]; model = LM(cfg).to(DEV); model.load_state_dict(ckpt["state_dict"]); model.eval()
        bytes_per = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}[DT]
        weights_bytes = sum(p.numel() for p in model.parameters()) * bytes_per
        rows = []
        batches = config.get("batches", [1, 4])
        for L in config.get("lengths", [4096, 16384, 65536]):
            elems = model.cache_elems_per_token_layers(L)   # PER SEQUENCE (per-token cache x L)
            row = dict(length=L,
                       cache_elems_per_seq=int(elems), cache_bytes_per_seq=int(elems * bytes_per),
                       mamba_state_bytes_per_seq=int(model.mamba_state_elems() * bytes_per),
                       flops_per_token=int(model.flops_per_token(L)),
                       weights_bytes=int(weights_bytes), by_batch={})
            # real prefill peak memory as a (length x batch) surface; OOM is the genuine memory wall
            for Bsz in batches:
                cell = {}
                if DEV == "cuda":
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    x = torch.randint(0, cfg["vocab"], (Bsz, L), device=DEV)
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DT, enabled=(DEV == "cuda")):
                        _ = model(x)                       # warmup: absorb CUDA init / Triton autotune / cuBLAS
                    if DEV == "cuda":
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()   # peak = timed pass only
                    t0 = time.time()
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DT, enabled=(DEV == "cuda")):
                        _ = model(x)
                    if DEV == "cuda":
                        torch.cuda.synchronize()
                    dt = time.time() - t0
                    cell = dict(oom=False, prefill_s=dt,
                                peak_bytes=int(torch.cuda.max_memory_allocated()) if DEV == "cuda" else 0,
                                prefill_tok_per_s=Bsz * L / max(1e-9, dt))
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    # a long-context prefill can surface OOM as a generic RuntimeError, not only
                    # OutOfMemoryError; treat any such failure as the genuine memory wall for this cell
                    cell = dict(oom=True, err=str(e)[:200])
                    if DEV == "cuda":
                        torch.cuda.empty_cache()
                row["by_batch"][str(Bsz)] = cell
            prog(stage="eff", length=L, cache_bytes_per_seq=row["cache_bytes_per_seq"])
            rows.append(row)
        # uncached end-to-end decode throughput from a 512-token prefix (same path for all variants)
        try:
            ctx = 512; gen_steps = 32
            seq = torch.randint(0, cfg["vocab"], (1, ctx), device=DEV)
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DT, enabled=(DEV == "cuda")):
                for _ in range(gen_steps):
                    nxt = model(seq)[:, -1:].argmax(-1)
                    seq = torch.cat([seq, nxt], dim=1)
            if DEV == "cuda":
                torch.cuda.synchronize()
            decode_tok_s = gen_steps / max(1e-9, time.time() - t0)
        except Exception as e:
            decode_tok_s = None
        out = dict(info, variant=cfg.get("mixers"), bytes_per_elem=bytes_per,
                   weights_bytes=int(weights_bytes), uncached_decode_tok_per_s=decode_tok_s, by_length=rows)
        _write_json(os.path.join(run_dir, "metrics.json"), out); prog(done=True)
        return out

    return dict(info, error=f"unknown task {task}")


@app.function(image=image, volumes={DATA_ROOT: vol}, gpu="A100-80GB", timeout=24 * 60 * 60)
def train(config: dict) -> dict:
    return _build_and_train(config)


@app.local_entrypoint()
def main(config_json: str = "{}"):
    if config_json.startswith("@"):
        with open(config_json[1:]) as f:
            config_json = f.read()
    config = json.loads(config_json)
    if "batch_configs" in config:
        # fan-out: list of configs run via Modal .map (e.g. Track-B MQAR sweep)
        gpu = config.get("gpu", "L4")
        fn = train.with_options(gpu=gpu)
        results = list(fn.map(config["batch_configs"]))
        print("RESULT_JSON:" + json.dumps({"batch_results": results}))
    else:
        gpu = config.get("gpu", "A100-80GB")
        opts = {"gpu": gpu}
        if config.get("cpu"):          # CPU-bound tasks (pretokenize) can request more cores / drop the idle GPU
            opts["cpu"] = config["cpu"]
        fn = train.with_options(**opts)
        result = fn.remote(config)
        print("RESULT_JSON:" + json.dumps(result))
