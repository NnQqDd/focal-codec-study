"""Write FocalCodec 50 Hz token sequences for the audio in a pairs CSV.

``--pairs`` is a CSV with a source column, a target column, and an output
column, as in the evaluation ``metadata.csv`` files. Each cell is an audio
path. Absolute paths are used as given. Relative paths are resolved against
the CSV's directory, which is where ``00000.wav`` sits.

Each distinct audio file becomes one ``.txt`` in ``--output-dir``. The text
file is one line of space-separated token ids, the sequence ``sig_to_toks``
returns at 50 Hz (codebook 8192, one id per 20 ms). ``pairs.csv`` in that
directory repeats the input rows, with the three path columns pointing at
those ``.txt`` files, so ``analyze_content_alignment.py --pairs`` can read it.

    python focalcodec_tokenize.py --pairs metadata.csv --output-dir tokens/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


log = logging.getLogger("focalcodec-tokenize")

CHECKPOINT = "lucadellalib/focalcodec_50hz"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
SOURCE_COLUMNS = ("source", "source_file", "source_path")
TARGET_COLUMNS = ("target", "target_file", "target_path")
OUTPUT_COLUMNS = ("output", "output_file", "output_path")
ROLES = ("source", "target", "output")


def load_mono(path, sample_rate):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:1]
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0).contiguous()
    if wav.numel() == 0:
        raise SystemExit(f"{path} is empty after resampling to {sample_rate} Hz")
    return wav


def load_codec(checkpoint, device):
    import focalcodec

    codec = focalcodec.FocalCodec.from_pretrained(checkpoint)
    codec.eval()
    for param in codec.parameters():
        param.requires_grad_(False)
    codec.to(device)
    hop = int(codec.encoder.downsample_factor)
    sample_rate = int(codec.sample_rate_input)
    rate = sample_rate / hop
    if abs(rate - 50.0) > 0.1:
        log.warning("token rate is %.3f Hz; this script expects the 50 Hz checkpoint", rate)
    return codec, hop, sample_rate


def parse_args():
    p = argparse.ArgumentParser(
        description="Tokenize the source, target, and output audio listed in a pairs CSV."
    )
    p.add_argument("--pairs", required=True,
                   help="CSV with a source column, a target column, and an "
                        "output column. Each cell is an audio path")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the .txt token files and pairs.csv")
    p.add_argument("--checkpoint", default=CHECKPOINT,
                   help=f"FocalCodec checkpoint (default: {CHECKPOINT})")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pick_column(columns, names):
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in names:
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None


def resolve_audio(value, meta_dir):
    """Absolute cells are full paths. Relative cells hang off the CSV directory."""
    raw = "" if pd.isna(value) else str(value).strip()
    if raw.lower() in {"", "nan", "none"}:
        raise SystemExit("empty audio path")
    given = Path(raw).expanduser()
    if given.is_absolute():
        candidates = [given]
    else:
        candidates = [meta_dir / given, meta_dir / given.name]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    tried = ", ".join(str(cand) for cand in candidates)
    raise SystemExit(f"audio not found: {raw} (tried {tried})")


def load_pairs(pairs_path):
    path = Path(pairs_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"pairs not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"{path} has no rows")
    columns = {
        "source": pick_column(df.columns, SOURCE_COLUMNS),
        "target": pick_column(df.columns, TARGET_COLUMNS),
        "output": pick_column(df.columns, OUTPUT_COLUMNS),
    }
    missing = [role for role, col in columns.items() if col is None]
    if missing:
        raise SystemExit(
            f"{path} has columns {list(df.columns)}. Missing {missing}. "
            f"Source is one of {list(SOURCE_COLUMNS)}, "
            f"target is one of {list(TARGET_COLUMNS)}, "
            f"output is one of {list(OUTPUT_COLUMNS)}"
        )
    meta_dir = path.parent
    rows = []
    for index, record in df.iterrows():
        audio = {
            role: resolve_audio(record[columns[role]], meta_dir)
            for role in ROLES
        }
        for role, file in audio.items():
            if file.suffix.lower() not in AUDIO_SUFFIXES:
                raise SystemExit(
                    f"{path} row {index} {role} is not audio: {file.name}"
                )
        extra = {
            col: record[col] for col in df.columns if col not in columns.values()
        }
        rows.append({"audio": audio, "extra": extra, "columns": columns})
    return rows, columns


def assign_names(rows):
    """One ``.txt`` name per distinct audio path. Stems that collide get a suffix."""
    names = {}
    used = {}
    for row in rows:
        for path in row["audio"].values():
            if path in names:
                continue
            stem = path.stem
            name = f"{stem}.txt"
            n = 2
            while name in used and used[name] != path:
                name = f"{stem}-{n}.txt"
                n += 1
            used[name] = path
            names[path] = name
    return names


@torch.inference_mode()
def tokenize_waveform(codec, wav, device, vocab):
    """Return the ``(frames,)`` int64 token ids of one mono waveform."""
    toks = codec.sig_to_toks(wav.to(device).unsqueeze(0))
    toks = toks.squeeze(0).detach().to("cpu").long()
    if toks.ndim != 1 or toks.numel() < 1:
        raise RuntimeError(f"expected a token sequence, got {tuple(toks.shape)}")
    low = int(toks.min())
    high = int(toks.max())
    if low < 0 or high >= vocab:
        raise RuntimeError(
            f"token id outside [0, {vocab}), got min {low} max {high}"
        )
    return toks


def write_tokens(path, toks):
    path.write_text(" ".join(str(int(token)) for token in toks.tolist()) + "\n",
                    encoding="utf-8")


def write_pairs_csv(rows, names, columns, path):
    """Repeat the input rows with the three audio columns replaced by ``.txt`` names."""
    records = []
    for row in rows:
        record = dict(row["extra"])
        for role in ROLES:
            record[columns[role]] = names[row["audio"][role]]
        records.append(record)
    # Path columns first, then the remaining input columns in their original order.
    fieldnames = [columns[role] for role in ROLES]
    extra_names = []
    for row in rows:
        for key in row["extra"]:
            if key not in extra_names:
                extra_names.append(key)
    frame = pd.DataFrame(records, columns=fieldnames + extra_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"output dir is not a directory: {output_dir}")
    rows, columns = load_pairs(args.pairs)
    names = assign_names(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    codec, hop, sample_rate = load_codec(args.checkpoint, device)
    vocab = int(codec.quantizer.codebook_size)
    log.info(
        "checkpoint %s on %s · %d ids · hop %d at %d Hz · %d audio file(s), %d pair(s)",
        args.checkpoint, device, vocab, hop, sample_rate, len(names), len(rows),
    )

    for audio, name in tqdm(names.items(), desc="tokenize", dynamic_ncols=True):
        wav = load_mono(audio, sample_rate)
        try:
            toks = tokenize_waveform(codec, wav, device, vocab)
        except RuntimeError as exc:
            raise SystemExit(f"{audio}: {exc}") from exc
        destination = output_dir / name
        write_tokens(destination, toks)
        log.info("%s -> %s (%d tokens)", audio.name, name, int(toks.numel()))

    pairs_csv = output_dir / "pairs.csv"
    write_pairs_csv(rows, names, columns, pairs_csv)
    log.info("wrote %d token file(s) and %s", len(names), pairs_csv)


if __name__ == "__main__":
    main()
