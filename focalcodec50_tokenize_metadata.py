"""Write FocalCodec 50 Hz token sequences for the audio listed in a metadata CSV.

``--metadata`` is a CSV like ``metadatas/test.csv``. It must have a
``filepath`` column. Every other column is copied into the result CSV.

Each distinct audio file becomes one ``.txt`` in ``--output-dir``. The
text file is one line of space-separated token ids, the sequence
``sig_to_toks`` returns at 50 Hz (codebook 8192, one id per 20 ms). The
result CSV has the input's name and columns. Its ``filepath`` cells name
those ``.txt`` files.

    python focalcodec50_tokenize_metadata.py \
        --metadata metadatas/test.csv \
        --output-dir tokens/

An absolute ``filepath`` is used as given. A relative one is resolved
against the CSV's directory.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


log = logging.getLogger("focalcodec50-tokenize-metadata")

CHECKPOINT = "lucadellalib/focalcodec_50hz"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
FILEPATH_COLUMN = "filepath"


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
        description="Tokenize the filepath column of a metadata CSV with FocalCodec 50 Hz."
    )
    p.add_argument("--metadata", required=True,
                   help="CSV with a filepath column. Other columns are copied through")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the .txt token files and the result CSV")
    p.add_argument("--checkpoint", default=CHECKPOINT,
                   help=f"FocalCodec checkpoint (default: {CHECKPOINT})")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def filepath_column(columns):
    for column in columns:
        if str(column).strip().lower() == FILEPATH_COLUMN:
            return column
    return None


def resolve_audio(value, meta_dir, row_index):
    """Absolute cells are full paths. Relative cells hang off the CSV directory."""
    raw = "" if pd.isna(value) else str(value).strip()
    if raw.lower() in {"", "nan", "none"}:
        raise SystemExit(f"row {row_index} has an empty filepath")
    given = Path(raw).expanduser()
    path = given if given.is_absolute() else meta_dir / given
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"row {row_index} audio not found: {path}")
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise SystemExit(f"row {row_index} is not audio: {path.name}")
    return path


def load_metadata(metadata_path):
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"metadata not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"{path} has no rows")
    column = filepath_column(df.columns)
    if column is None:
        raise SystemExit(
            f"{path} has columns {list(df.columns)}. A {FILEPATH_COLUMN} column is required"
        )
    audios = [
        resolve_audio(value, path.parent, index)
        for index, value in df[column].items()
    ]
    return df, column, audios


def assign_names(audios):
    """One ``.txt`` name per distinct audio path. Stems that collide get a suffix."""
    names = {}
    used = {}
    for audio in audios:
        if audio in names:
            continue
        name = f"{audio.stem}.txt"
        n = 2
        while name in used and used[name] != audio:
            name = f"{audio.stem}-{n}.txt"
            n += 1
        used[name] = audio
        names[audio] = name
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


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"output dir is not a directory: {output_dir}")
    metadata_path = Path(args.metadata).expanduser().resolve()
    frame, column, audios = load_metadata(metadata_path)
    names = assign_names(audios)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    codec, hop, sample_rate = load_codec(args.checkpoint, device)
    vocab = int(codec.quantizer.codebook_size)
    log.info(
        "checkpoint %s on %s · %d ids · hop %d at %d Hz · %d audio file(s), %d row(s)",
        args.checkpoint, device, vocab, hop, sample_rate, len(names), len(audios),
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

    result = frame.copy()
    result[column] = [names[audio] for audio in audios]
    result_csv = output_dir / metadata_path.name
    result.to_csv(result_csv, index=False)
    log.info("wrote %d token file(s) and %s", len(names), result_csv)


if __name__ == "__main__":
    main()
