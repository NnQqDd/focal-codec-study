"""Speaker clustering of mean-pooled FocalCodec token embeddings.

``--metadata`` is a CSV with a ``filepath`` column and a ``speaker``
column. Each filepath is a ``.txt`` of token ids, as written by
``focalcodec50_tokenize_metadata.py``. Each id selects a row of
``--embeddings``. The utterance vector is the mean of those rows.
K-means is fit with one cluster per speaker. Silhouette measures
whether those averages form speaker-shaped groups. B-cubed precision,
recall, and F1 measure agreement with the speaker labels. The F1 is
the B-cubed result.

    python analyze_speaker.py \
        --metadata tokens/test.csv \
        --embeddings embeddings/focalcodec50.npy \
        --result-dir speaker-clustering

An absolute filepath is used as given. A relative one is resolved against
the CSV's directory. When the CSV has a ``duration`` column, that value
is the seconds written to the assignment table. Otherwise seconds are the
token count divided by 50.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import bcubed


log = logging.getLogger("focalcodec-speaker")

PATH_COLUMNS = ("filepath", "path")
SPEAKER_COLUMNS = ("speaker",)
DURATION_COLUMNS = ("duration",)
FRAME_HZ = 50.0

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
    p.add_argument("--metadata", required=True,
                   help="CSV with a filepath column of .txt token files and a speaker column")
    p.add_argument("--embeddings", required=True,
                   help="Numpy array of shape (vocab, dim), one row per token id")
    p.add_argument(
        "--cluster-config",
        default=str(Path(__file__).with_name("cluster.yaml")),
        help="YAML with a kmeans key (default: cluster.yaml beside this script)",
    )
    p.add_argument("--result-dir", required=True,
                   help="Directory for assignments.csv and summary.json")
    p.add_argument("--limit", type=int, default=0,
                   help="Embed only the first N rows (0 = all)")
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


def pick_column(columns, names):
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in names:
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None


def resolve_token(value, meta_dir):
    """Absolute cells are full paths. Relative cells hang off the CSV directory."""
    raw = "" if pd.isna(value) else str(value).strip()
    if raw.lower() in {"", "nan", "none"}:
        raise SystemExit("empty filepath")
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
    """Rows of one metadata CSV."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"metadata not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"{path} has no rows")
    path_col = pick_column(df.columns, PATH_COLUMNS)
    spk_col = pick_column(df.columns, SPEAKER_COLUMNS)
    if path_col is None or spk_col is None:
        raise SystemExit(
            f"{path} has columns {list(df.columns)}. Need one of "
            f"{list(PATH_COLUMNS)} and one of {list(SPEAKER_COLUMNS)}"
        )
    duration_col = pick_column(df.columns, DURATION_COLUMNS)
    if limit and limit > 0:
        df = df.iloc[:limit]
    rows = []
    for index, record in df.iterrows():
        speaker = record[spk_col]
        if pd.isna(speaker) or str(speaker).strip().lower() in {"", "nan", "none"}:
            raise SystemExit(f"{path} row {index} has an empty speaker")
        seconds = None
        if duration_col is not None and not pd.isna(record[duration_col]):
            seconds = float(record[duration_col])
        rows.append({
            "path": str(resolve_token(record[path_col], path.parent)),
            "speaker": speaker_label(speaker),
            "seconds": seconds,
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
    on hard assignments. The library call is what gets reported.
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


def embed_rows(table, rows):
    vocab = int(table.shape[0])
    vectors = []
    for row in tqdm(rows, desc="embed", dynamic_ncols=True):
        ids = read_tokens(row["path"], vocab)
        frames = table[ids].astype(np.float64, copy=False)
        row["n_frames"] = int(frames.shape[0])
        if row["seconds"] is None:
            row["seconds"] = row["n_frames"] / FRAME_HZ
        vectors.append(frames.mean(axis=0))
    stacked = np.stack(vectors)
    if not np.isfinite(stacked).all():
        raise SystemExit("non-finite utterance embedding")
    return stacked


def run_metadata(path, rows, save_dir, table, cfg):
    speakers = [row["speaker"] for row in rows]
    n_speakers = len(set(speakers))
    log.info("%s: %d utterances, %d speakers", path.name, len(rows), n_speakers)
    embeddings = embed_rows(table, rows)
    features = prepare_features(embeddings, cfg["kmeans"]["normalize"])
    clusters = fit_kmeans(features, n_speakers, cfg["kmeans"])
    library = bcubed_library(speakers, clusters)
    fast = bcubed_hard(speakers, clusters)
    if max(abs(library[i] - fast[i]) for i in range(3)) > 1e-6:
        raise RuntimeError(
            f"B-cubed mismatch on {path.name}: library {library} vs closed form {fast}"
        )
    precision, recall, f1 = library
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
    embed_dim = int(table.shape[1])
    summary = {
        "embeddings": cfg["embeddings"],
        "representation": "mean of embedding-table rows",
        "representation_detail": (
            "Each frame is the row of the embedding table selected by its "
            "token id. The utterance vector is the mean of those rows."
        ),
        "embed_dim": embed_dim,
        "metadata": str(path),
        "normalize": bool(cfg["kmeans"]["normalize"]),
        "kmeans": cfg["kmeans"],
        **metrics,
    }
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print()
    print(
        f"{path.name}  ·  {metrics['n_utterances']} utterances  ·  "
        f"{metrics['n_speakers']} speakers  ·  mean of {embed_dim}-D frames"
    )
    for label, key in (
        ("silhouette", "silhouette"),
        ("B-cubed precision", "bcubed_precision"),
        ("B-cubed recall", "bcubed_recall"),
        ("B-cubed F1", "bcubed_f1"),
    ):
        print(f"  {label:<28} {_fmt(metrics[key])}")
    print()
    print(f"wrote {assignments_path}")
    print(f"wrote {summary_path}")
    return summary


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg, config_path = load_cluster_config(args.cluster_config)
    metadata_path = Path(args.metadata).expanduser().resolve()
    rows = load_rows(metadata_path, args.limit)
    table = load_embedding_table(args.embeddings)
    cfg = {
        "embeddings": str(Path(args.embeddings).expanduser().resolve()),
        "kmeans": cfg["kmeans"],
    }
    save_dir = Path(args.result_dir).expanduser().resolve()
    log.info("cluster config %s", config_path)
    log.info(
        "normalize=%s, silhouette metric=%s",
        cfg["kmeans"]["normalize"], cfg["kmeans"]["silhouette_metric"],
    )
    log.info(
        "embeddings %s · %d ids · %d-D",
        cfg["embeddings"], table.shape[0], table.shape[1],
    )
    run_metadata(metadata_path, rows, save_dir, table, cfg)


if __name__ == "__main__":
    main()
