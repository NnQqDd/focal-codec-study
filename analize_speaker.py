"""Speaker clustering of mean-pooled FocalCodec 50 Hz embeddings.

Each utterance is tokenized with FocalCodec 50 Hz, lifted to 256-D by
``decompressor.in_proj`` (one code at a time, no focal block), and averaged
over time. K-means is fit with one cluster per speaker in the metadata.
Silhouette measures whether those averages form speaker-shaped groups.
B-cubed precision, recall, and F1 measure agreement with the speaker
labels. The F1 is the B-cubed result. A speaker shuffle is the null: it
keeps the speaker sizes and destroys the link to the clusters.

    python analize_speaker.py \
        --metadata-dir /path/to/metadatas_or_one.csv \
        --cluster-config cluster.yaml

A directory is every ``*.csv`` sitting directly in it. A csv path is that
one file. Each file needs a path column (``path`` or ``filepath``) and a
``speaker`` column, as in ``speech_v1_dev.csv``. Files without those
columns are skipped when the argument is a directory.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import bcubed

from focalcodec_embed import (
    CHECKPOINT,
    EMBED_DIM,
    average_embedding,
    embed_waveform,
    load_codec,
    load_mono,
    pick_column,
    resolve_audio,
)


log = logging.getLogger("focalcodec-speaker")

PATH_COLUMNS = ("path", "filepath")
SPEAKER_COLUMNS = ("speaker",)
N_SHUFFLES = 32

# n_clusters is the speaker count, so it is not a config key.
DEFAULTS = {
    "kmeans": {
        "normalize": True,
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "algorithm": "lloyd",
        "random_state": 0,
        "silhouette_metric": "euclidean",
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Cluster mean-pooled FocalCodec 50 Hz embeddings by speaker."
    )
    p.add_argument(
        "--metadata-dir", required=True,
        help="Directory of metadata CSVs, or one CSV. Columns: path (or "
             "filepath) and speaker",
    )
    p.add_argument(
        "--cluster-config",
        default=str(Path(__file__).with_name("cluster.yaml")),
        help="YAML with a kmeans key (default: cluster.yaml beside this script)",
    )
    p.add_argument("--save-dir", default=None,
                   help="Output directory (default: ./speaker-clustering). "
                        "A directory of CSVs writes one subdirectory per file")
    p.add_argument("--checkpoint", default=CHECKPOINT,
                   help=f"FocalCodec checkpoint (default: {CHECKPOINT})")
    p.add_argument("--limit", type=int, default=0,
                   help="Embed only the first N rows of each CSV (0 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _is_bool(value):
    return isinstance(value, bool)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expect(section, key, value, kind):
    ok = {
        "bool": _is_bool,
        "int": _is_int,
        "number": _is_number,
        "str": lambda v: isinstance(v, str),
    }[kind](value)
    if not ok:
        raise SystemExit(f"{section}.{key} must be {kind}, got {value!r}")


def load_cluster_config(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"cluster config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must be a mapping with a kmeans key")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise SystemExit(
            f"{path} has unknown keys {sorted(unknown)}. Expected kmeans"
        )
    cfg = {}
    for section, defaults in DEFAULTS.items():
        over = raw.get(section, {})
        if over is None:
            over = {}
        if not isinstance(over, dict):
            raise SystemExit(f"{section} must be a mapping")
        bad = set(over) - set(defaults)
        if bad:
            raise SystemExit(f"unknown {section} keys: {sorted(bad)}")
        merged = dict(defaults)
        merged.update(over)
        cfg[section] = merged
    _validate_config(cfg)
    return cfg, path


def _validate_config(cfg):
    km = cfg["kmeans"]
    _expect("kmeans", "normalize", km["normalize"], "bool")
    _expect("kmeans", "max_iter", km["max_iter"], "int")
    _expect("kmeans", "tol", km["tol"], "number")
    _expect("kmeans", "algorithm", km["algorithm"], "str")
    _expect("kmeans", "random_state", km["random_state"], "int")
    _expect("kmeans", "silhouette_metric", km["silhouette_metric"], "str")
    n_init = km["n_init"]
    if isinstance(n_init, str):
        if n_init != "auto":
            raise SystemExit("kmeans.n_init must be a positive integer or auto")
    elif _is_int(n_init):
        if n_init < 1:
            raise SystemExit("kmeans.n_init must be >= 1")
    else:
        raise SystemExit("kmeans.n_init must be a positive integer or auto")
    if km["max_iter"] < 1:
        raise SystemExit("kmeans.max_iter must be >= 1")
    if km["tol"] <= 0:
        raise SystemExit("kmeans.tol must be positive")
    if km["algorithm"] not in {"lloyd", "elkan"}:
        raise SystemExit("kmeans.algorithm must be lloyd or elkan")
    if not str(km["silhouette_metric"]).strip():
        raise SystemExit("kmeans.silhouette_metric is empty")


def metadata_files(metadata_dir):
    path = Path(metadata_dir).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise SystemExit(f"metadata is not a csv: {path}")
        return [path], False
    if not path.is_dir():
        raise SystemExit(f"metadata not found: {path}")
    files = sorted(
        child for child in path.iterdir()
        if child.is_file() and child.suffix.lower() == ".csv"
    )
    if not files:
        raise SystemExit(f"no csv files in {path}")
    return files, True


def speaker_label(value):
    """String speaker id. Integer-valued floats stay integers, so 1272 and 1272.0 match."""
    if isinstance(value, (bool, np.bool_)):
        raise SystemExit("speaker cell is a boolean")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def load_rows(path, limit):
    """Rows of one metadata CSV, or None when the columns are absent."""
    df = pd.read_csv(path)
    path_col = pick_column(df.columns, PATH_COLUMNS)
    spk_col = pick_column(df.columns, SPEAKER_COLUMNS)
    if path_col is None or spk_col is None:
        return None
    if limit and limit > 0:
        df = df.iloc[:limit]
    if df.empty:
        raise SystemExit(f"{path} has no rows")
    rows = []
    for index, record in df.iterrows():
        speaker = record[spk_col]
        if pd.isna(speaker) or str(speaker).strip().lower() in {"", "nan", "none"}:
            raise SystemExit(f"{path} row {index} has an empty speaker")
        rows.append({
            "path": str(resolve_audio(record[path_col], path.parent)),
            "speaker": speaker_label(speaker),
        })
    return rows


def prepare_features(embeddings, normalize):
    x = np.asarray(embeddings, dtype=np.float64)
    if not np.isfinite(x).all():
        raise SystemExit("non-finite utterance embedding")
    if normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise SystemExit("an averaged embedding has zero norm")
        x = x / norms
    return x


def bcubed_hard(speakers, clusters):
    """B-cubed precision, recall, and F1 for one label and one cluster per item.

    This is the same quantity as ``bcubed.precision`` / ``bcubed.recall``
    on hard assignments. The library call is what gets reported; this form
    is the shuffle null, which would otherwise be quadratic in pure Python.
    """
    speakers = np.asarray(speakers)
    clusters = np.asarray(clusters)
    n = int(speakers.shape[0])
    _spk, spk_inv = np.unique(speakers, return_inverse=True)
    _cls, cls_inv = np.unique(clusters, return_inverse=True)
    table = np.zeros((int(spk_inv.max()) + 1, int(cls_inv.max()) + 1), dtype=np.float64)
    np.add.at(table, (spk_inv, cls_inv), 1.0)
    size_s = table.sum(axis=1)
    size_c = table.sum(axis=0)
    precision = float((table * table / size_c).sum() / n)
    recall = float((table * table / size_s[:, None]).sum() / n)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def bcubed_library(speakers, clusters):
    ldict = {str(i): {str(speaker)} for i, speaker in enumerate(speakers)}
    cdict = {str(i): {str(cluster)} for i, cluster in enumerate(clusters)}
    precision = float(bcubed.precision(cdict, ldict))
    recall = float(bcubed.recall(cdict, ldict))
    f1 = float(bcubed.fscore(precision, recall))
    return precision, recall, f1


def shuffled_bcubed(speakers, clusters, n_shuffles, seed):
    rng = np.random.default_rng(int(seed))
    speakers = np.asarray(speakers)
    recall_acc = 0.0
    f1_acc = 0.0
    for _ in range(n_shuffles):
        _precision, recall, f1 = bcubed_hard(rng.permutation(speakers), clusters)
        recall_acc += recall
        f1_acc += f1
    return recall_acc / n_shuffles, f1_acc / n_shuffles


def fit_kmeans(features, n_clusters, km):
    if n_clusters < 1 or n_clusters > features.shape[0]:
        raise SystemExit(
            f"need 1..n_utterances clusters, got {n_clusters} for {features.shape[0]} rows"
        )
    model = KMeans(
        n_clusters=n_clusters,
        n_init=km["n_init"],
        max_iter=km["max_iter"],
        tol=km["tol"],
        algorithm=km["algorithm"],
        random_state=km["random_state"],
    )
    return model.fit_predict(features)


def silhouette(features, clusters, metric):
    n_labels = len(set(int(c) for c in clusters))
    if n_labels < 2 or n_labels >= len(clusters):
        return None
    try:
        return float(silhouette_score(features, clusters, metric=metric))
    except ValueError as exc:
        log.warning("silhouette undefined: %s", exc)
        return None


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _round(value):
    value = _finite_or_none(value)
    if value is None:
        return None
    return round(value, 6)


def _fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def embed_rows(codec, proj, rows, sample_rate, device):
    vectors = []
    for row in tqdm(rows, desc="embed", dynamic_ncols=True):
        wav = load_mono(row["path"], sample_rate)
        try:
            emb, _codes, _toks = embed_waveform(codec, proj, wav, device)
            vec = average_embedding(emb).detach().float().cpu().numpy()
        except RuntimeError as exc:
            raise SystemExit(f"{row['path']}: {exc}") from exc
        row["n_frames"] = int(emb.shape[0])
        row["seconds"] = float(wav.shape[-1]) / sample_rate
        vectors.append(np.asarray(vec, dtype=np.float64))
    return np.stack(vectors)


def run_metadata(path, rows, save_dir, codec, proj, sample_rate, device, cfg):
    speakers = [row["speaker"] for row in rows]
    n_speakers = len(set(speakers))
    log.info("%s: %d utterances, %d speakers", path.name, len(rows), n_speakers)
    embeddings = embed_rows(codec, proj, rows, sample_rate, device)
    features = prepare_features(embeddings, cfg["kmeans"]["normalize"])
    clusters = fit_kmeans(features, n_speakers, cfg["kmeans"])
    library = bcubed_library(speakers, clusters)
    fast = bcubed_hard(speakers, clusters)
    if max(abs(library[i] - fast[i]) for i in range(3)) > 1e-6:
        raise RuntimeError(
            f"B-cubed mismatch on {path.name}: library {library} vs closed form {fast}"
        )
    precision, recall, f1 = library
    recall_shuf, f1_shuf = shuffled_bcubed(
        speakers, clusters, N_SHUFFLES, cfg["kmeans"]["random_state"],
    )
    sil = silhouette(features, clusters, cfg["kmeans"]["silhouette_metric"])
    sizes = np.bincount(clusters)
    sizes = sizes[sizes > 0]
    metrics = {
        "n_utterances": len(rows),
        "n_speakers": n_speakers,
        "n_clusters": n_speakers,
        "silhouette": _round(sil),
        "bcubed_precision": _round(precision),
        "bcubed_recall": _round(recall),
        "bcubed_f1": _round(f1),
        "bcubed_recall_shuffled": _round(recall_shuf),
        "bcubed_f1_shuffled": _round(f1_shuf),
        "n_shuffles": N_SHUFFLES,
        "cluster_size_min": int(sizes.min()),
        "cluster_size_max": int(sizes.max()),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame({
        "path": [row["path"] for row in rows],
        "speaker": speakers,
        "cluster": clusters.astype(int),
        "n_frames": [row["n_frames"] for row in rows],
        "seconds": [row["seconds"] for row in rows],
    })
    assignments_path = save_dir / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    summary = {
        "checkpoint": cfg["checkpoint"],
        "representation": "mean of decompressor.in_proj",
        "representation_detail": (
            "Each frame is a 13-D binary-spherical code lifted by "
            "Linear(13, 256) on its own. Focal blocks are not used. "
            "The utterance vector is the mean of those 256-D frames."
        ),
        "embed_dim": EMBED_DIM,
        "metadata": str(path),
        "normalize": bool(cfg["kmeans"]["normalize"]),
        "kmeans": cfg["kmeans"],
        **metrics,
        "reading": (
            "Across {n} utterances and {k} speakers, silhouette is {sil}. "
            "B-cubed precision is {precision}, recall is {recall}, and F1 is {f1}, "
            "against a speaker-shuffle recall of {recall_shuf} and F1 of {f1_shuf}. "
            "A silhouette near 1 and a B-cubed F1 well above the shuffle mean the "
            "averaged 256-D vector separates these speakers. A silhouette near 0 and "
            "an F1 near the shuffle mean the average does not carry speaker identity."
        ).format(
            n=metrics["n_utterances"],
            k=metrics["n_speakers"],
            sil=_fmt(metrics["silhouette"]),
            precision=_fmt(metrics["bcubed_precision"]),
            recall=_fmt(metrics["bcubed_recall"]),
            f1=_fmt(metrics["bcubed_f1"]),
            recall_shuf=_fmt(metrics["bcubed_recall_shuffled"]),
            f1_shuf=_fmt(metrics["bcubed_f1_shuffled"]),
        ),
    }
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print()
    print(
        f"{path.name}  ·  {metrics['n_utterances']} utterances  ·  "
        f"{metrics['n_speakers']} speakers  ·  mean of {EMBED_DIM}-D frames"
    )
    for label, key in (
        ("silhouette", "silhouette"),
        ("B-cubed precision", "bcubed_precision"),
        ("B-cubed recall", "bcubed_recall"),
        ("B-cubed F1", "bcubed_f1"),
        ("B-cubed recall, shuffled", "bcubed_recall_shuffled"),
        ("B-cubed F1, shuffled", "bcubed_f1_shuffled"),
    ):
        print(f"  {label:<28} {_fmt(metrics[key])}")
    print()
    print(summary["reading"])
    print()
    print(f"wrote {assignments_path}")
    print(f"wrote {summary_path}")
    return summary


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg, config_path = load_cluster_config(args.cluster_config)
    cfg = {
        "checkpoint": args.checkpoint,
        "kmeans": cfg["kmeans"],
    }
    files, is_dir = metadata_files(args.metadata_dir)
    loaded = []
    for path in files:
        rows = load_rows(path, args.limit)
        if rows is None:
            if is_dir:
                log.info("skip %s: no path and speaker columns", path.name)
                continue
            raise SystemExit(
                f"{path} needs a path column {list(PATH_COLUMNS)} and a speaker column"
            )
        loaded.append((path, rows))
    if not loaded:
        raise SystemExit(
            f"no metadata with columns {list(PATH_COLUMNS)} and {list(SPEAKER_COLUMNS)}"
        )

    save_root = (
        Path(args.save_dir).expanduser().resolve()
        if args.save_dir
        else Path.cwd() / "speaker-clustering"
    )
    multiple = len(loaded) > 1
    log.info("cluster config %s", config_path)
    log.info(
        "%d metadata file(s), normalize=%s, silhouette metric=%s",
        len(loaded), cfg["kmeans"]["normalize"], cfg["kmeans"]["silhouette_metric"],
    )

    device = torch.device(args.device)
    codec, proj, _hop, sample_rate = load_codec(args.checkpoint, device)
    log.info(
        "checkpoint %s on %s · embedding Linear(%d, %d) · mean over frames",
        args.checkpoint, device, proj.in_features, proj.out_features,
    )
    for path, rows in loaded:
        out = save_root / path.stem if multiple else save_root
        run_metadata(path, rows, out, codec, proj, sample_rate, device, cfg)


if __name__ == "__main__":
    main()
