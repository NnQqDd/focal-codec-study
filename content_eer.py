"""Content equal-error rate for one alignment result directory.

``--content-result-dir`` is enough. ``summary.json`` records the embedding
table, and ``pairs.csv`` records the source and output token files. A
target score is the cosine of frames at the same time. An impostor score
is the cosine after a time shuffle of the output. The equal-error rate is
the threshold where the false-accept rate equals the false-reject rate.
Lower is better. The rate is computed inside each pair, then averaged.

    python content_eer.py --content-result-dir results/second_mdlm_alignment
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tqdm import tqdm

from analyze_content import embed_ids, load_embedding_table, match_tokens, read_tokens


log = logging.getLogger("content-eer")


def parse_args():
    p = argparse.ArgumentParser(
        description="Content equal-error rate from an alignment result directory."
    )
    p.add_argument("--content-result-dir", required=True,
                   help="Directory with summary.json and pairs.csv from analyze_content.py")
    return p.parse_args()


def equal_error_rate(target, impostor):
    """Threshold where the false-accept rate equals the false-reject rate."""
    target = np.asarray(target, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if target.size < 1 or impostor.size < 1:
        raise RuntimeError("need target and impostor scores")
    labels = np.concatenate([np.ones(target.size), np.zeros(impostor.size)])
    scores = np.concatenate([target, impostor])
    fpr, tpr, _thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    cross = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if cross.size == 0:
        j = int(np.argmin(np.abs(diff)))
        return float((fpr[j] + fnr[j]) / 2.0)
    i = int(cross[0])
    d0 = float(diff[i])
    d1 = float(diff[i + 1])
    if d0 == d1:
        return float(fpr[i])
    weight = d0 / (d0 - d1)
    return float(fpr[i] + weight * (fpr[i + 1] - fpr[i]))


def unit_rows(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def pair_eer(table, source_ids, output_ids, n_shuffles, seed):
    src = unit_rows(embed_ids(table, source_ids))
    out = unit_rows(embed_ids(table, output_ids))
    target = (src * out).sum(axis=1)
    rng = np.random.default_rng(int(seed))
    impostor = np.empty(n_shuffles * target.shape[0], dtype=np.float64)
    length = target.shape[0]
    for draw in range(n_shuffles):
        perm = rng.permutation(length)
        start = draw * length
        impostor[start:start + length] = (src * out[perm]).sum(axis=1)
    return equal_error_rate(target, impostor)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result_dir = Path(args.content_result_dir).expanduser().resolve()
    summary_path = result_dir / "summary.json"
    pairs_path = result_dir / "pairs.csv"
    if not summary_path.is_file() or not pairs_path.is_file():
        raise SystemExit(f"{result_dir} needs summary.json and pairs.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    embeddings = summary.get("embeddings")
    if not embeddings:
        raise SystemExit(f"{summary_path} has no embeddings path")
    n_shuffles = int(summary.get("n_shuffles", 32))
    if n_shuffles < 1:
        raise SystemExit(f"{summary_path} has n_shuffles {n_shuffles}")
    seed = int(summary.get("seed", 0))
    align = summary.get("align", "crop")

    table = load_embedding_table(embeddings)
    vocab = int(table.shape[0])
    pairs = pd.read_csv(pairs_path)
    required = {"source", "output", "index"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise SystemExit(f"{pairs_path} is missing columns {missing}")
    if "align" not in pairs.columns:
        pairs["align"] = align

    rates = []
    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="eer", dynamic_ncols=True):
        source_ids = read_tokens(row.source, vocab)
        output_ids = read_tokens(row.output, vocab)
        source_ids, output_ids = match_tokens(source_ids, output_ids, row.align)
        rate = pair_eer(table, source_ids, output_ids, n_shuffles, seed + int(row.index))
        rates.append({
            "index": int(row.index),
            "source": row.source,
            "output": row.output,
            "eer": rate,
            "n_frames": int(source_ids.shape[0]),
        })

    values = np.array([item["eer"] for item in rates], dtype=np.float64)
    report = {
        "content_result_dir": str(result_dir),
        "embeddings": str(Path(embeddings).expanduser().resolve()),
        "n_pairs": len(rates),
        "n_shuffles": n_shuffles,
        "eer_mean": round(float(values.mean()), 6),
        "eer_std": round(float(values.std()), 6),
        "eer_median": round(float(np.median(values)), 6),
    }
    report_path = result_dir / "eer.json"
    pairs_eer_path = result_dir / "eer.csv"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rates).to_csv(pairs_eer_path, index=False)
    print(
        f"{result_dir.name}  ·  {report['n_pairs']} pairs  ·  "
        f"content EER {report['eer_mean']:.3%}  ·  std {report['eer_std']:.3%}"
    )
    print(f"wrote {report_path}")
    print(f"wrote {pairs_eer_path}")


if __name__ == "__main__":
    main()
