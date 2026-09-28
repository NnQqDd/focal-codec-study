"""Frame-level content alignment of FocalCodec 50 Hz tokens.

Reproduces Figure 5 of Tu et al., Findings of EMNLP 2025 (O_O-VC,
https://aclanthology.org/2025.findings-emnlp.879.pdf). That figure asks
whether two recordings of the same words line up in a frame embedding:

  (a) cosine similarity between every source frame and every output frame;
  (b) the top-1 path — for each source frame, the output frame it matches
      best. A content code puts that path on the diagonal.

The paper uses HuBERT-large frame embeddings. Here each frame is a
FocalCodec 50 Hz token. The token itself is a 13-D binary code on the
unit hypersphere (codebook 8192, one code per 20 ms). The decompressor's
first layer, ``in_proj``, is Linear(13, 256) and is applied to each code
on its own, before any focal block mixes neighboring frames. That 256-D
vector is the embedding this script compares. Later decompressor layers
are left unused because they pool temporal context into the frame.
Loading that embedding is ``focalcodec_embed``, shared with
``analize_speaker.py``.

Each pair is resampled to 16 kHz and forced to one length, then trimmed
to a whole number of codec hops. ``--align crop`` (default) keeps the
shared prefix of the two waveforms. ``--align pad`` zero-pads the shorter
one at the end. Utterance-mean subtraction is not applied: a constant
offset is exactly the speaker component this measurement is trying to see.

The null for "this is content" is a time shuffle of one side. A shuffle
keeps the token multiset of the utterance and destroys the alignment.
Speaker identity is constant across frames, so it survives the shuffle.
Content does not. The gap between the aligned diagonal and the shuffled
diagonal is the quantity to read.

    python analyze_content_alignment.py --metadata-path metadata.csv \
        --save-dir alignment

The metadata CSV needs a source column (``source_audio``, ``source``, or
``source_path``) and an output column (``output_audio``, ``output``,
``converted``, ``converted_audio``, ``output_path``, or ``converted_path``).
An absolute cell is used as that full path. A relative cell is resolved
against the metadata file's directory and nowhere else.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection
from tqdm import tqdm

from focalcodec_embed import (
    CHECKPOINT,
    EMBED_DIM,
    embed_waveform,
    load_codec,
    load_mono,
    pick_column,
    resolve_audio,
)


log = logging.getLogger("focalcodec-alignment")

# Fixed color scale of Figure 5: cosine from 0 (blue / purple) to 1 (yellow).
VMIN = 0.0
VMAX = 1.0
# Above this, the heatmap image is strided. Metrics always use every frame.
PLOT_MAX_FRAMES = 2500

SOURCE_COLUMNS = ("source_audio", "source", "source_path")
OUTPUT_COLUMNS = (
    "output_audio",
    "output",
    "converted",
    "converted_audio",
    "output_path",
    "converted_path",
)

METRIC_KEYS = (
    "diagonal_cosine",
    "off_diagonal_cosine",
    "diagonal_margin",
    "shuffled_diagonal_cosine",
    "content_gap",
    "top1_cosine",
    "top1_excess",
    "aligned_near_top1_rate",
    "top1_exact_rate",
    "top1_within_40ms",
    "top1_within_100ms",
    "top1_mean_abs_lag_ms",
    "top1_backward_rate",
    "path_correlation",
    "row_entropy",
    "reverse_top1_within_100ms",
    "reverse_path_correlation",
    "token_match_rate",
    "token_match_shuffled",
    "code_diagonal_cosine",
    "code_shuffled_cosine",
    "code_content_gap",
    "source_self_offdiag",
    "output_self_offdiag",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="FocalCodec 50 Hz content alignment, in the style of "
                    "O_O-VC Figure 5 (Tu et al., EMNLP 2025 Findings)."
    )
    p.add_argument("--metadata-path", required=True,
                   help="CSV with a source column and an output column")
    p.add_argument("--save-dir", default=None,
                   help="Where to write pairs.csv, summary.json, summary.png, "
                        "and figures/ (default: <metadata dir>/focalcodec_alignment)")
    p.add_argument("--checkpoint", default=CHECKPOINT,
                   help=f"FocalCodec checkpoint (default: {CHECKPOINT})")
    p.add_argument("--align", choices=("crop", "pad"), default="crop",
                   help="crop: keep the shared prefix (default). "
                        "pad: zero-pad the shorter waveform at the end")
    p.add_argument("--max-figures", type=int, default=-1,
                   help="Pair figures to write, in CSV order. -1 writes every pair")
    p.add_argument("--n-shuffles", type=int, default=32,
                   help="Time-shuffle draws for the content-gap null")
    p.add_argument("--entropy-temperature", type=float, default=0.1,
                   help="Softmax temperature on cosine for row entropy")
    p.add_argument("--block-size", type=int, default=512,
                   help="Query frames per cosine block")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="Analyze only the first N pairs (0 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_pairs(metadata_path, limit):
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"metadata not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"{path} has no rows")
    src_col = pick_column(df.columns, SOURCE_COLUMNS)
    out_col = pick_column(df.columns, OUTPUT_COLUMNS)
    if src_col is None or out_col is None:
        raise SystemExit(
            f"{path} has columns {list(df.columns)}. Need one of "
            f"{list(SOURCE_COLUMNS)} and one of {list(OUTPUT_COLUMNS)}"
        )
    if limit and limit > 0:
        df = df.iloc[:limit]
    pairs = []
    for index, row in df.iterrows():
        pairs.append({
            "index": int(index) if isinstance(index, (int, np.integer)) else len(pairs),
            "source": str(resolve_audio(row[src_col], path.parent)),
            "output": str(resolve_audio(row[out_col], path.parent)),
        })
    # CSV index may restart; keep a stable 0-based position too.
    for position, pair in enumerate(pairs):
        pair["position"] = position
    return pairs, src_col, out_col


def match_length(source, output, policy, hop):
    """Return two equal-length waveforms on the codec hop grid.

    ``crop`` drops the tail of the longer waveform. ``pad`` extends the
    shorter one with zeros. Both are then trimmed to a multiple of ``hop``
    so the encoder emits the same number of frames for each.
    """
    n_src = int(source.shape[-1])
    n_out = int(output.shape[-1])
    if policy == "crop":
        n = min(n_src, n_out)
        source = source[:n]
        output = output[:n]
    elif policy == "pad":
        n = max(n_src, n_out)
        source = F.pad(source, (0, n - n_src))
        output = F.pad(output, (0, n - n_out))
    else:
        raise ValueError(policy)
    n = (int(source.shape[-1]) // hop) * hop
    if n < hop:
        raise SystemExit(
            f"pair is shorter than one codec frame after {policy} "
            f"(hop={hop} samples)"
        )
    return source[:n].contiguous(), output[:n].contiguous()


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def off_diagonal_mean(left, right):
    """Mean cosine over i != j. Rows of ``left`` and ``right`` are L2-normalized."""
    t = left.shape[0]
    total = torch.dot(left.sum(dim=0), right.sum(dim=0))
    diag = (left * right).sum()
    return (total - diag) / (t * (t - 1))


def self_off_diagonal(feats):
    t = feats.shape[0]
    total = torch.dot(feats.sum(dim=0), feats.sum(dim=0))
    return (total - t) / (t * (t - 1))


def row_match(query, key, tau, block):
    """Per-query top-1 over keys, plus mean normalized row entropy.

    Both arguments are L2-normalized, shape (T, D). Entropy is
    ``H(softmax(cosine / tau)) / log T``, so 0 is a one-frame peak and 1
    is a flat row.
    """
    t = query.shape[0]
    vals = torch.empty(t, device=query.device)
    idxs = torch.empty(t, dtype=torch.long, device=query.device)
    ent_sum = query.new_zeros(())
    log_t = math.log(t)
    positions = torch.arange(t, device=query.device)
    for start in range(0, t, block):
        sim = query[start:start + block] @ key.T
        b = sim.shape[0]
        row_pos = torch.arange(start, start + b, device=query.device)
        # A repeated token ties at cosine 1. Plain argmax returns the earliest
        # frame and pulls the path off the diagonal. Among numerical ties, take
        # the frame closest in time, which credits the aligned frame when it
        # is itself a best match.
        peak = sim.amax(dim=-1, keepdim=True)
        nearest = sim >= peak - 1e-5
        dist = (positions[None, :] - row_pos[:, None]).abs()
        dist = dist.masked_fill(~nearest, t + 1)
        idx = dist.argmin(dim=-1)
        val = sim.gather(1, idx[:, None]).squeeze(1)
        vals[start:start + b] = val
        idxs[start:start + b] = idx
        z = sim / tau
        z = z - z.max(dim=-1, keepdim=True).values
        prob = z.exp()
        prob = prob / prob.sum(dim=-1, keepdim=True)
        ent = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1) / log_t
        ent_sum = ent_sum + ent.sum()
    return vals, idxs, ent_sum / t


def shuffled_diagonal(left, right, n_shuffles, generator):
    """Mean diagonal cosine after permuting ``right``. Rows are normalized."""
    t = left.shape[0]
    acc = left.new_zeros(())
    for _ in range(n_shuffles):
        perm = torch.randperm(t, generator=generator)
        acc = acc + (left * right[perm.to(left.device)]).sum(dim=-1).mean()
    return acc / n_shuffles


def shuffled_token_match(source_toks, output_toks, n_shuffles, generator):
    t = source_toks.shape[0]
    acc = 0.0
    for _ in range(n_shuffles):
        perm = torch.randperm(t, generator=generator)
        acc += float((source_toks == output_toks[perm]).float().mean())
    return acc / n_shuffles


def path_stats(idxs, frame_seconds):
    """How close the top-1 path sits to the identity line."""
    t = idxs.shape[0]
    frames = torch.arange(t, device=idxs.device)
    lag_frames = (idxs - frames).abs().float()
    lag_seconds = lag_frames * frame_seconds
    exact = float((lag_frames == 0).float().mean())
    lag_ms = float(lag_seconds.mean() * 1000.0)
    if t >= 2:
        backward = float((idxs[1:] < idxs[:-1]).float().mean())
    else:
        backward = float("nan")
    corr = pearson(frames.detach().cpu().numpy(), idxs.detach().float().cpu().numpy())
    return {
        "exact": exact,
        "lag_ms": lag_ms,
        "backward": backward,
        "correlation": corr,
        "within_40ms": float((lag_seconds <= 0.040 + 1e-6).float().mean()),
        "within_100ms": float((lag_seconds <= 0.100 + 1e-6).float().mean()),
    }


def measure_pair(emb_s, emb_o, code_s, code_o, tok_s, tok_o, frame_seconds,
                 n_shuffles, tau, block, seed):
    if emb_s.shape[0] != emb_o.shape[0]:
        raise RuntimeError(
            f"frame counts differ after length matching ({emb_s.shape[0]} vs {emb_o.shape[0]})"
        )
    t = emb_s.shape[0]
    if t < 2:
        raise SystemExit("need at least two frames to measure off-diagonal cosine")

    src = F.normalize(emb_s, dim=-1)
    out = F.normalize(emb_o, dim=-1)
    # Codes are already ±1/sqrt(dim) on the unit sphere. Normalize anyway
    # so a future checkpoint that drifts off the sphere still yields cosine.
    code_src = F.normalize(code_s, dim=-1)
    code_out = F.normalize(code_o, dim=-1)

    diag = (src * out).sum(dim=-1)
    diagonal = float(diag.mean())
    off = float(off_diagonal_mean(src, out))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shuffled = float(shuffled_diagonal(src, out, n_shuffles, generator))
    code_diag = float((code_src * code_out).sum(dim=-1).mean())
    code_shuf = float(shuffled_diagonal(code_src, code_out, n_shuffles, generator))

    top_val, top_idx, entropy = row_match(src, out, tau, block)
    _rev_val, rev_idx, _rev_ent = row_match(out, src, tau, block)
    forward = path_stats(top_idx, frame_seconds)
    reverse = path_stats(rev_idx, frame_seconds)
    # Fraction of frames whose aligned partner is within 0.05 cosine of the
    # best partner anywhere. This stays high when the diagonal is a ridge
    # even if a repeated phone elsewhere wins the strict top-1.
    near_top1 = float((diag >= top_val - 0.05).float().mean())

    token_match = float((tok_s == tok_o).float().mean())
    token_shuf = shuffled_token_match(tok_s.cpu(), tok_o.cpu(), n_shuffles, generator)

    return {
        "n_frames": int(t),
        "diagonal_cosine": diagonal,
        "off_diagonal_cosine": off,
        "diagonal_margin": diagonal - off,
        "shuffled_diagonal_cosine": shuffled,
        "content_gap": diagonal - shuffled,
        "top1_cosine": float(top_val.mean()),
        "top1_excess": float(top_val.mean()) - diagonal,
        "aligned_near_top1_rate": near_top1,
        "top1_exact_rate": forward["exact"],
        "top1_within_40ms": forward["within_40ms"],
        "top1_within_100ms": forward["within_100ms"],
        "top1_mean_abs_lag_ms": forward["lag_ms"],
        "top1_backward_rate": forward["backward"],
        "path_correlation": forward["correlation"],
        "row_entropy": float(entropy),
        "reverse_top1_within_100ms": reverse["within_100ms"],
        "reverse_path_correlation": reverse["correlation"],
        "token_match_rate": token_match,
        "token_match_shuffled": token_shuf,
        "code_diagonal_cosine": code_diag,
        "code_shuffled_cosine": code_shuf,
        "code_content_gap": code_diag - code_shuf,
        "source_self_offdiag": float(self_off_diagonal(src)),
        "output_self_offdiag": float(self_off_diagonal(out)),
        "source_n_unique_tokens": int(tok_s.unique().numel()),
        "output_n_unique_tokens": int(tok_o.unique().numel()),
        "_top_idx": top_idx.detach().cpu(),
        "_top_val": top_val.detach().cpu(),
        "_src": src.detach(),
        "_out": out.detach(),
    }


def cosine_matrix(src, out, step):
    return (src[::step] @ out[::step].T).detach().float().cpu().numpy()


def _short_name(path, limit=48):
    name = Path(path).name
    if len(name) <= limit:
        return name
    return name[: limit - 1] + "…"


def save_pair_figure(pair, metrics, frame_hz, path):
    src = metrics["_src"]
    out = metrics["_out"]
    top_idx = metrics["_top_idx"].numpy()
    top_val = metrics["_top_val"].numpy()
    t = src.shape[0]
    step = max(1, math.ceil(t / PLOT_MAX_FRAMES))
    sim = cosine_matrix(src, out, step)
    plotted = sim.shape[0]
    # Strided heatmaps stay on the original frame index, same axis as the path.
    extent = (
        -0.5 * step,
        (plotted - 0.5) * step,
        -0.5 * step,
        (plotted - 0.5) * step,
    )

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 9.6))

    ax = axes[0]
    image = ax.imshow(
        sim,
        origin="lower",
        aspect="equal",
        cmap="jet",
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
        extent=extent,
    )
    ax.set_title("(a) Cosine similarity matrix between source and output audio")
    ax.set_xlabel("Source audio")
    ax.set_ylabel("Output audio")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")

    ax = axes[1]
    frames = np.arange(t, dtype=np.float64)
    ax.plot(frames, frames, color="0.45", ls="--", lw=0.9, label="identity", zorder=1)
    if t >= 2:
        points = np.stack([frames, top_idx.astype(np.float64)], axis=1)
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = 0.5 * (top_val[:-1] + top_val[1:])
        collection = LineCollection(
            segments, cmap="viridis", norm=plt.Normalize(VMIN, VMAX), linewidth=1.6, zorder=2,
        )
        collection.set_array(np.clip(colors, VMIN, VMAX))
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine similarity")
    ax.scatter(frames, top_idx, c=np.clip(top_val, VMIN, VMAX), cmap="viridis",
               vmin=VMIN, vmax=VMAX, s=6, zorder=3, linewidths=0)
    ax.set_xlim(-0.5, t - 0.5)
    ax.set_ylim(-0.5, t - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(b) Top-1 cosine similarity alignment path")
    ax.set_xlabel("Source audio")
    ax.set_ylabel("Output audio")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    stride_note = f", plotted every {step} frames" if step > 1 else ""
    fig.suptitle(
        f"{_short_name(pair['source'])}  →  {_short_name(pair['output'])}\n"
        f"FocalCodec 50 Hz · decompressor.in_proj · 13-D → 256-D"
        f" · {frame_hz:.0f} Hz{stride_note}",
        fontsize=11,
    )
    def _shown(key, percent=False):
        value = metrics[key]
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        if percent:
            return f"{float(value):.0%}"
        return f"{float(value):.2f}"

    fig.text(
        0.5, 0.005,
        "diagonal {d}    shuffle {s}    gap {g}    within 100 ms {w}    "
        "token match {m}    path r {r}".format(
            d=_shown("diagonal_cosine"),
            s=_shown("shuffled_diagonal_cosine"),
            g=_shown("content_gap"),
            w=_shown("top1_within_100ms", percent=True),
            m=_shown("token_match_rate", percent=True),
            r=_shown("path_correlation"),
        ),
        ha="center", va="bottom", fontsize=8, color="0.25",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_summary_figure(rows, path):
    diag = np.array([r["diagonal_cosine"] for r in rows], dtype=np.float64)
    off = np.array([r["off_diagonal_cosine"] for r in rows], dtype=np.float64)
    shuf = np.array([r["shuffled_diagonal_cosine"] for r in rows], dtype=np.float64)
    gap = np.array([r["content_gap"] for r in rows], dtype=np.float64)
    within = np.array([r["top1_within_100ms"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    ax = axes[0]
    series = [diag, off, shuf]
    if len(rows) >= 2:
        ax.boxplot(series, widths=0.55, showfliers=False,
                   medianprops={"color": "0.15"})
    rng = np.random.default_rng(0)
    for i, vals in enumerate(series, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, zorder=3,
                   color="#1f4e79", alpha=0.85)
    ax.set_xticks([1, 2, 3], ["aligned\ndiagonal", "off-diagonal", "time-shuffled\ndiagonal"])
    ax.set_ylabel("Cosine similarity of 256-D token embeddings")
    ax.set_title("Aligned frames against the time-shuffle null")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="0.85", lw=0.6)

    ax = axes[1]
    ax.scatter(gap, within, s=28, color="#1f4e79", zorder=3)
    ax.set_xlabel("Content gap (aligned cosine − shuffled cosine)")
    ax.set_ylabel("Top-1 match within 100 ms")
    ax.set_title("One point per source / output pair")
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(0.0, color="0.85", lw=0.6)
    finite_gap = gap[np.isfinite(gap)]
    if finite_gap.size:
        pad = 0.05
        lo = min(float(finite_gap.min()) - pad, -pad)
        hi = max(float(finite_gap.max()) + pad, pad)
        ax.set_xlim(lo, hi)

    fig.suptitle("FocalCodec 50 Hz · does the token track aligned content?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _finite_mean(values):
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return None
    return float(np.nanmean(arr))


def _round(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, 6)


def summarize(rows, args, hop, sample_rate, src_col, out_col, codebook_size):
    means = {key: _round(_finite_mean(r[key] for r in rows)) for key in METRIC_KEYS}
    stds = {}
    for key in METRIC_KEYS:
        arr = np.array([r[key] for r in rows], dtype=np.float64)
        stds[key] = _round(float(np.nanstd(arr))) if np.isfinite(arr).any() else None
    chance = 1.0 / float(codebook_size)
    means["token_match_chance_uniform"] = _round(chance)
    reading = (
        "Across {n} pair(s), aligned frames have mean cosine {diag}, "
        "time-shuffled frames {shuf} (gap {gap}), and off-diagonal frames {off}. "
        "The top-1 output frame falls within 100 ms of the source frame on "
        "{within} of frames, and the top-1 path correlates with the identity "
        "line at {corr}. Exact token match is {match}, against {match_shuf} "
        "after a time shuffle and {chance} if codes were drawn uniformly from "
        "the {codebook}-entry codebook. "
        "A wide gap over the shuffle, a top-1 path along the diagonal, and "
        "token matches above the shuffle mean that the 256-D code tracks the "
        "content of that frame across the two recordings. A gap near zero and "
        "a wandering path mean the code does not follow aligned content, which "
        "is the pattern of a code that is constant across the utterance, such "
        "as speaker identity."
    ).format(
        n=len(rows),
        diag=_fmt(means["diagonal_cosine"]),
        shuf=_fmt(means["shuffled_diagonal_cosine"]),
        gap=_fmt(means["content_gap"]),
        off=_fmt(means["off_diagonal_cosine"]),
        within=_fmt_pct(means["top1_within_100ms"]),
        corr=_fmt(means["path_correlation"]),
        match=_fmt_pct(means["token_match_rate"]),
        match_shuf=_fmt_pct(means["token_match_shuffled"]),
        chance=_fmt_pct(chance),
        codebook=codebook_size,
    )
    return {
        "checkpoint": args.checkpoint,
        "representation": "decompressor.in_proj",
        "representation_detail": (
            "Linear(13, 256) applied independently to each binary-spherical "
            "code. Focal blocks, which mix neighboring frames, are not used."
        ),
        "embed_dim": EMBED_DIM,
        "code_dim": 13,
        "codebook_size": int(codebook_size),
        "sample_rate": int(sample_rate),
        "hop": int(hop),
        "frame_rate_hz": sample_rate / hop,
        "align": args.align,
        "n_shuffles": int(args.n_shuffles),
        "entropy_temperature": float(args.entropy_temperature),
        "source_column": src_col,
        "output_column": out_col,
        "n_pairs": len(rows),
        "mean": means,
        "std": stds,
        "reading": reading,
    }


def _fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_pct(value):
    if value is None:
        return "n/a"
    pct = 100.0 * float(value)
    if pct != 0.0 and abs(pct) < 0.05:
        return f"{pct:.3f}%"
    return f"{pct:.1f}%"


def print_summary(summary):
    mean = summary["mean"]
    print()
    print(f"{summary['n_pairs']} pair(s)  ·  {summary['frame_rate_hz']:.2f} Hz  ·  "
          f"align={summary['align']}  ·  {summary['representation']}")
    lines = (
        ("aligned cosine", "diagonal_cosine"),
        ("off-diagonal cosine", "off_diagonal_cosine"),
        ("shuffled cosine", "shuffled_diagonal_cosine"),
        ("content gap", "content_gap"),
        ("top-1 excess over diagonal", "top1_excess"),
        ("aligned frame near top-1", "aligned_near_top1_rate"),
        ("top-1 within 100 ms", "top1_within_100ms"),
        ("top-1 |lag| (ms)", "top1_mean_abs_lag_ms"),
        ("path correlation", "path_correlation"),
        ("row entropy", "row_entropy"),
        ("token match", "token_match_rate"),
        ("token match, shuffled", "token_match_shuffled"),
        ("13-D code content gap", "code_content_gap"),
        ("within-utterance off-diag (source)", "source_self_offdiag"),
    )
    for label, key in lines:
        value = mean[key]
        if key in {
            "top1_within_100ms", "aligned_near_top1_rate",
            "token_match_rate", "token_match_shuffled",
        }:
            shown = _fmt_pct(value)
        else:
            shown = _fmt(value)
        print(f"  {label:<36} {shown}")
    print()
    print(summary["reading"])


def drop_private(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.n_shuffles < 1:
        raise SystemExit("--n-shuffles must be >= 1")
    if args.entropy_temperature <= 0:
        raise SystemExit("--entropy-temperature must be positive")

    metadata_path = Path(args.metadata_path).expanduser().resolve()
    save_dir = (
        Path(args.save_dir).expanduser().resolve()
        if args.save_dir
        else metadata_path.parent / "focalcodec_alignment"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = save_dir / "figures"

    pairs, src_col, out_col = load_pairs(metadata_path, args.limit)
    log.info("%d pair(s) from %s (%s, %s)", len(pairs), metadata_path, src_col, out_col)

    device = torch.device(args.device)
    codec, proj, hop, sample_rate = load_codec(args.checkpoint, device)
    frame_seconds = hop / sample_rate
    frame_hz = sample_rate / hop
    codebook_size = int(codec.quantizer.codebook_size)
    log.info(
        "checkpoint %s on %s · %d-D code, codebook %d, %.2f Hz · embedding Linear(%d, %d)",
        args.checkpoint, device, int(codec.quantizer.dim), codebook_size, frame_hz,
        proj.in_features, proj.out_features,
    )

    rows = []
    n_figures = len(pairs) if args.max_figures < 0 else min(args.max_figures, len(pairs))
    for pair in tqdm(pairs, desc="align", dynamic_ncols=True):
        source = load_mono(pair["source"], sample_rate)
        output = load_mono(pair["output"], sample_rate)
        n_src = int(source.shape[-1])
        n_out = int(output.shape[-1])
        source, output = match_length(source, output, args.align, hop)
        emb_s, code_s, tok_s = embed_waveform(codec, proj, source, device)
        emb_o, code_o, tok_o = embed_waveform(codec, proj, output, device)
        metrics = measure_pair(
            emb_s, emb_o, code_s, code_o, tok_s, tok_o, frame_seconds,
            args.n_shuffles, args.entropy_temperature, args.block_size,
            args.seed + pair["position"],
        )
        record = {
            "index": pair["position"],
            "source": pair["source"],
            "output": pair["output"],
            "align": args.align,
            "source_seconds": n_src / sample_rate,
            "output_seconds": n_out / sample_rate,
            "matched_seconds": int(source.shape[-1]) / sample_rate,
            **drop_private(metrics),
        }
        if pair["position"] < n_figures:
            figure_path = figure_dir / f"{pair['position']:04d}.png"
            save_pair_figure(pair, metrics, frame_hz, figure_path)
            record["figure"] = str(figure_path)
        else:
            record["figure"] = ""
        # Drop the tensors before the next pair.
        del metrics
        rows.append(record)

    pairs_path = save_dir / "pairs.csv"
    pd.DataFrame(rows).to_csv(pairs_path, index=False)
    summary = summarize(rows, args, hop, sample_rate, src_col, out_col, codebook_size)
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_summary_figure(rows, save_dir / "summary.png")
    print_summary(summary)
    print()
    print(f"wrote {pairs_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {save_dir / 'summary.png'}")
    if n_figures:
        print(f"wrote {n_figures} pair figure(s) in {figure_dir}")


if __name__ == "__main__":
    main()
