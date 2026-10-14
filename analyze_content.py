"""Frame-level content alignment of FocalCodec token embeddings.

Reproduces the cosine-similarity matrix of Figure 5 in Tu et al.,
Findings of EMNLP 2025 (O_O-VC,
https://aclanthology.org/2025.findings-emnlp.879.pdf). The matrix is the
cosine between every source frame and every output frame. A content code
puts the bright ridge on the diagonal.

Each frame vector is one row of ``--embeddings``, indexed by the token id
in a ``.txt`` file. ``--pairs`` is the CSV that names each pair: a source
column, a target column, and an output column. Each cell is the path of
that side's token file. ``focalcodec_tokenize.py`` writes those files.

``--align crop`` (default) keeps the shared prefix of the two token
sequences. ``--align pad`` extends the shorter sequence with empty frames.
An empty frame is the zero vector, so its cosine is 0, and its id matches
no token. Utterance-mean subtraction is left off: a constant offset is
the speaker component this measurement is trying to see.

The null for "this is content" is a time shuffle of one side. A shuffle
keeps the token multiset of the utterance and destroys the alignment.
Speaker identity is constant across frames, so it survives the shuffle.
Content does not. The gap between the aligned diagonal and the shuffled
diagonal is the quantity to read.

    python analyze_content.py --pairs pairs.csv \
        --embeddings embeddings/focalcodec50.npy \
        --result-dir alignment

The CSV needs one source column (``source``, ``source_file``, or
``source_path``), one target column (``target``, ``target_file``, or
``target_path``), and one output column (``output``, ``output_file``, or
``output_path``). The cosine matrix compares source with output. An
absolute cell is used as that full path. A relative cell is resolved
against the CSV's directory and nowhere else. Durations are counted on
the 50 Hz token grid.
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
from tqdm import tqdm


log = logging.getLogger("focalcodec-alignment")

# Fixed color scale of Figure 5: cosine from 0 (blue / purple) to 1 (yellow).
VMIN = 0.0
VMAX = 1.0
# Above this, the heatmap image is strided. Metrics always use every frame.
PLOT_MAX_FRAMES = 2500
FRAME_HZ = 50.0

SOURCE_COLUMNS = ("source", "source_file", "source_path")
TARGET_COLUMNS = ("target", "target_file", "target_path")
OUTPUT_COLUMNS = ("output", "output_file", "output_path")

METRIC_KEYS = (
    "diagonal_cosine",
    "off_diagonal_cosine",
    "diagonal_margin",
    "shuffled_diagonal_cosine",
    "content_gap",
    "token_match_rate",
    "token_match_shuffled",
    "source_self_offdiag",
    "output_self_offdiag",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="FocalCodec token cosine-similarity matrix, in the style "
                    "of O_O-VC Figure 5 (Tu et al., EMNLP 2025 Findings)."
    )
    p.add_argument("--pairs", required=True,
                   help="CSV with a source column, a target column, and an "
                        "output column. Each cell is a .txt token file")
    p.add_argument("--embeddings", required=True,
                   help="Numpy array of shape (vocab, dim), one row per token id")
    p.add_argument("--result-dir", required=True,
                   help="Directory for pairs.csv, summary.json, and figures/")
    p.add_argument("--align", choices=("crop", "pad"), default="crop",
                   help="crop: keep the shared token prefix (default). "
                        "pad: extend the shorter sequence with empty frames")
    p.add_argument("--max-figures", type=int, default=-1,
                   help="Pair figures to write, in CSV order. -1 writes every pair")
    p.add_argument("--n-shuffles", type=int, default=32,
                   help="Time-shuffle draws for the content-gap null")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="Analyze only the first N pairs (0 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pick_column(columns, names):
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in names:
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None


def token_path(value, meta_dir):
    """Absolute cells are full paths. Relative cells hang off ``meta_dir`` only."""
    raw = "" if pd.isna(value) else str(value).strip()
    if raw.lower() in {"", "nan", "none"}:
        raise SystemExit("empty token path")
    given = Path(raw).expanduser()
    path = given if given.is_absolute() else meta_dir / given
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"token file not found: {path}")
    return path


def load_embedding_table(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"embeddings not found: {path}")
    try:
        table = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read embeddings {path}: {exc}") from exc
    if not isinstance(table, np.ndarray) or table.ndim != 2:
        raise SystemExit(f"{path} must be a 2-D array, got shape {getattr(table, 'shape', None)}")
    if table.shape[0] < 1 or table.shape[1] < 1:
        raise SystemExit(f"{path} is empty")
    table = np.asarray(table, dtype=np.float32)
    if not np.isfinite(table).all():
        raise SystemExit(f"{path} has non-finite values")
    return table


def read_tokens(path, vocab):
    """One whitespace-separated sequence of token ids."""
    text = Path(path).read_text(encoding="utf-8")
    parts = text.split()
    if not parts:
        raise SystemExit(f"{path} has no token ids")
    try:
        ids = np.array([int(part) for part in parts], dtype=np.int64)
    except ValueError as exc:
        raise SystemExit(f"{path} has a non-integer token: {exc}") from exc
    low = int(ids.min())
    high = int(ids.max())
    if low < 0 or high >= vocab:
        raise SystemExit(f"{path} has a token id outside [0, {vocab})")
    return ids


def load_pairs(pairs_path, limit):
    path = Path(pairs_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"pairs not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"{path} has no rows")
    src_col = pick_column(df.columns, SOURCE_COLUMNS)
    tgt_col = pick_column(df.columns, TARGET_COLUMNS)
    out_col = pick_column(df.columns, OUTPUT_COLUMNS)
    missing = [
        label for label, col in (
            ("source", src_col), ("target", tgt_col), ("output", out_col),
        ) if col is None
    ]
    if missing:
        raise SystemExit(
            f"{path} has columns {list(df.columns)}. Missing {missing}. "
            f"Source is one of {list(SOURCE_COLUMNS)}, "
            f"target is one of {list(TARGET_COLUMNS)}, "
            f"output is one of {list(OUTPUT_COLUMNS)}"
        )
    if limit and limit > 0:
        df = df.iloc[:limit]
    pairs = []
    for index, row in df.iterrows():
        source = token_path(row[src_col], path.parent)
        target = token_path(row[tgt_col], path.parent)
        output = token_path(row[out_col], path.parent)
        pairs.append({
            "index": int(index) if isinstance(index, (int, np.integer)) else len(pairs),
            "source": str(source),
            "target": str(target),
            "output": str(output),
        })
    for position, pair in enumerate(pairs):
        pair["position"] = position
    return pairs, src_col, tgt_col, out_col


def _pad_ids(ids, length):
    if ids.shape[0] == length:
        return ids
    padded = np.full(length, -1, dtype=np.int64)
    padded[:ids.shape[0]] = ids
    return padded


def match_tokens(source, output, policy):
    """Return two equal-length token sequences.

    ``crop`` keeps the shared prefix. ``pad`` extends the shorter sequence
    with ``-1``, an empty frame that is not a codebook id.
    """
    n_src = int(source.shape[0])
    n_out = int(output.shape[0])
    if policy == "crop":
        n = min(n_src, n_out)
        source = source[:n]
        output = output[:n]
    elif policy == "pad":
        n = max(n_src, n_out)
        source = _pad_ids(source, n)
        output = _pad_ids(output, n)
    else:
        raise ValueError(policy)
    if n < 2:
        raise SystemExit(f"need at least two frames after {policy}")
    return source, output


def embed_ids(table, ids):
    """Rows of ``table``. Id ``-1`` is an empty frame, the zero vector."""
    emb = np.zeros((ids.shape[0], table.shape[1]), dtype=np.float32)
    real = ids >= 0
    if real.any():
        emb[real] = table[ids[real]]
    return emb


def unit_rows(x):
    """L2-normalize rows. A zero row stays zero, so its cosine is 0."""
    norms = x.norm(dim=-1, keepdim=True)
    return x / norms.clamp_min(1e-12)


def off_diagonal_mean(left, right):
    """Mean cosine over i != j. Rows of ``left`` and ``right`` are L2-normalized."""
    t = left.shape[0]
    total = torch.dot(left.sum(dim=0), right.sum(dim=0))
    diag = (left * right).sum()
    return (total - diag) / (t * (t - 1))


def self_off_diagonal(feats):
    """Mean cosine over i != j. A zero row has self-cosine 0, so subtract the real diagonal."""
    t = feats.shape[0]
    total = torch.dot(feats.sum(dim=0), feats.sum(dim=0))
    diag = (feats * feats).sum()
    return (total - diag) / (t * (t - 1))


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
    source_toks = source_toks.cpu()
    output_toks = output_toks.cpu()
    acc = 0.0
    for _ in range(n_shuffles):
        perm = torch.randperm(t, generator=generator)
        acc += float((source_toks == output_toks[perm]).float().mean())
    return acc / n_shuffles


def measure_pair(emb_s, emb_o, tok_s, tok_o, n_shuffles, seed):
    if emb_s.shape[0] != emb_o.shape[0]:
        raise RuntimeError(
            f"frame counts differ after length matching ({emb_s.shape[0]} vs {emb_o.shape[0]})"
        )
    t = emb_s.shape[0]
    if t < 2:
        raise SystemExit("need at least two frames to measure off-diagonal cosine")

    src = unit_rows(emb_s)
    out = unit_rows(emb_o)
    diag = (src * out).sum(dim=-1)
    diagonal = float(diag.mean())
    off = float(off_diagonal_mean(src, out))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shuffled = float(shuffled_diagonal(src, out, n_shuffles, generator))
    token_match = float((tok_s == tok_o).float().mean())
    token_shuf = shuffled_token_match(tok_s, tok_o, n_shuffles, generator)
    return {
        "n_frames": int(t),
        "diagonal_cosine": diagonal,
        "off_diagonal_cosine": off,
        "diagonal_margin": diagonal - off,
        "shuffled_diagonal_cosine": shuffled,
        "content_gap": diagonal - shuffled,
        "token_match_rate": token_match,
        "token_match_shuffled": token_shuf,
        "source_self_offdiag": float(self_off_diagonal(src)),
        "output_self_offdiag": float(self_off_diagonal(out)),
        "source_n_unique_tokens": int(tok_s[tok_s >= 0].unique().numel()),
        "output_n_unique_tokens": int(tok_o[tok_o >= 0].unique().numel()),
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


def save_pair_figure(pair, metrics, path):
    src = metrics["_src"]
    out = metrics["_out"]
    t = src.shape[0]
    step = max(1, math.ceil(t / PLOT_MAX_FRAMES))
    sim = cosine_matrix(src, out, step)
    plotted = sim.shape[0]
    extent = (
        -0.5 * step,
        (plotted - 0.5) * step,
        -0.5 * step,
        (plotted - 0.5) * step,
    )

    fig, ax = plt.subplots(figsize=(6.3, 5.6))
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
    ax.set_title("Cosine similarity matrix between source and output")
    ax.set_xlabel("Source audio")
    ax.set_ylabel("Output audio")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")

    stride_note = f", plotted every {step} frames" if step > 1 else ""
    fig.suptitle(
        f"{_short_name(pair['source'])}  →  {_short_name(pair['output'])}\n"
        f"FocalCodec token embeddings · {FRAME_HZ:.0f} Hz{stride_note}",
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
        "diagonal {d}    shuffle {s}    gap {g}    token match {m}".format(
            d=_shown("diagonal_cosine"),
            s=_shown("shuffled_diagonal_cosine"),
            g=_shown("content_gap"),
            m=_shown("token_match_rate", percent=True),
        ),
        ha="center", va="bottom", fontsize=8, color="0.25",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
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


def summarize(rows, args, table, src_col, tgt_col, out_col):
    means = {key: _round(_finite_mean(r[key] for r in rows)) for key in METRIC_KEYS}
    stds = {}
    for key in METRIC_KEYS:
        arr = np.array([r[key] for r in rows], dtype=np.float64)
        stds[key] = _round(float(np.nanstd(arr))) if np.isfinite(arr).any() else None
    codebook_size = int(table.shape[0])
    means["token_match_chance_uniform"] = _round(1.0 / float(codebook_size))
    return {
        "embeddings": str(Path(args.embeddings).expanduser().resolve()),
        "pairs": str(Path(args.pairs).expanduser().resolve()),
        "representation": "embedding-table row",
        "representation_detail": (
            "Each frame is the row of the embedding table selected by its token id. "
            "The cosine matrix compares the source token file with the output token file."
        ),
        "embed_dim": int(table.shape[1]),
        "codebook_size": codebook_size,
        "frame_rate_hz": FRAME_HZ,
        "align": args.align,
        "n_shuffles": int(args.n_shuffles),
        "source_column": src_col,
        "target_column": tgt_col,
        "output_column": out_col,
        "n_pairs": len(rows),
        "mean": means,
        "std": stds,
    }


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
        ("token match", "token_match_rate"),
        ("token match, shuffled", "token_match_shuffled"),
        ("within-utterance off-diag (source)", "source_self_offdiag"),
    )
    for label, key in lines:
        value = mean[key]
        if key in {"token_match_rate", "token_match_shuffled"}:
            shown = _fmt_pct(value)
        else:
            shown = _fmt(value)
        print(f"  {label:<36} {shown}")


def drop_private(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.n_shuffles < 1:
        raise SystemExit("--n-shuffles must be >= 1")

    table = load_embedding_table(args.embeddings)
    pairs_path = Path(args.pairs).expanduser().resolve()
    save_dir = Path(args.result_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = save_dir / "figures"

    pairs, src_col, tgt_col, out_col = load_pairs(pairs_path, args.limit)
    log.info(
        "%d pair(s) from %s (%s, %s, %s)",
        len(pairs), pairs_path, src_col, tgt_col, out_col,
    )
    log.info(
        "embeddings %s · %d ids · %d-D",
        Path(args.embeddings).expanduser().resolve(),
        table.shape[0], table.shape[1],
    )

    device = torch.device(args.device)
    vocab = int(table.shape[0])
    rows = []
    n_figures = len(pairs) if args.max_figures < 0 else min(args.max_figures, len(pairs))
    for pair in tqdm(pairs, desc="align", dynamic_ncols=True):
        source_ids = read_tokens(pair["source"], vocab)
        output_ids = read_tokens(pair["output"], vocab)
        n_src = int(source_ids.shape[0])
        n_out = int(output_ids.shape[0])
        source_ids, output_ids = match_tokens(source_ids, output_ids, args.align)
        emb_s = torch.as_tensor(embed_ids(table, source_ids), device=device)
        emb_o = torch.as_tensor(embed_ids(table, output_ids), device=device)
        tok_s = torch.as_tensor(source_ids, device=device)
        tok_o = torch.as_tensor(output_ids, device=device)
        metrics = measure_pair(
            emb_s, emb_o, tok_s, tok_o, args.n_shuffles, args.seed + pair["position"],
        )
        record = {
            "index": pair["position"],
            "source": pair["source"],
            "target": pair["target"],
            "output": pair["output"],
            "align": args.align,
            "source_seconds": n_src / FRAME_HZ,
            "output_seconds": n_out / FRAME_HZ,
            "matched_seconds": int(source_ids.shape[0]) / FRAME_HZ,
            **drop_private(metrics),
        }
        if pair["position"] < n_figures:
            figure_path = figure_dir / f"{pair['position']:04d}.png"
            save_pair_figure(pair, metrics, figure_path)
            record["figure"] = str(figure_path)
        else:
            record["figure"] = ""
        del metrics
        rows.append(record)

    results_path = save_dir / "pairs.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    summary = summarize(rows, args, table, src_col, tgt_col, out_col)
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print()
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    if n_figures:
        print(f"wrote {n_figures} pair figure(s) in {figure_dir}")


if __name__ == "__main__":
    main()
