"""FocalCodec 50 Hz loading and the 256-D frame embedding.

The 50 Hz token is a 13-D binary code on the unit hypersphere. The
decompressor's first layer, ``in_proj``, is Linear(13, 256) and runs on
each code alone, before any focal block mixes neighboring frames. Later
decompressor layers are left unused because they pool temporal context
into the frame. ``analyze_content_alignment.py`` compares these frame
vectors. ``analize_speaker.py`` averages them over an utterance.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchaudio


log = logging.getLogger(__name__)

CHECKPOINT = "lucadellalib/focalcodec_50hz"
EMBED_DIM = 256


def pick_column(columns, names):
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in names:
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None


def resolve_audio(value, meta_dir):
    """Absolute cells are full paths. Relative cells hang off ``meta_dir`` only."""
    raw = "" if pd.isna(value) else str(value).strip()
    if raw.lower() in {"", "nan", "none"}:
        raise SystemExit("empty audio path")
    given = Path(raw).expanduser()
    path = given if given.is_absolute() else meta_dir / given
    if not path.is_file():
        raise SystemExit(f"audio not found: {path}")
    return path.resolve()


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

    proj = codec.decompressor.in_proj
    code_dim = int(codec.quantizer.dim)
    if not isinstance(proj, nn.Linear) or proj.in_features != code_dim or proj.out_features != EMBED_DIM:
        raise SystemExit(
            "decompressor.in_proj is "
            f"Linear({getattr(proj, 'in_features', '?')}, {getattr(proj, 'out_features', '?')}). "
            f"FocalCodec 50 Hz maps each {code_dim}-D code through Linear({code_dim}, {EMBED_DIM}) "
            "before any temporal layer. Refusing a different projection."
        )
    hop = int(codec.encoder.downsample_factor)
    sample_rate = int(codec.sample_rate_input)
    rate = sample_rate / hop
    if abs(rate - 50.0) > 0.1:
        log.warning("token rate is %.3f Hz; this script expects the 50 Hz checkpoint", rate)
    return codec, proj, hop, sample_rate


@torch.inference_mode()
def embed_waveform(codec, proj, wav, device):
    """13-D code, discrete token, and the independent 256-D lift of each frame."""
    sig = wav.to(device).unsqueeze(0)
    codes = codec.sig_to_codes(sig)
    emb = proj(codes)
    toks = codec.codes_to_toks(codes)
    emb = emb.squeeze(0).float()
    codes = codes.squeeze(0).float()
    toks = toks.squeeze(0).long()
    if emb.ndim != 2 or emb.shape[-1] != EMBED_DIM:
        raise RuntimeError(f"expected (frames, {EMBED_DIM}) embeddings, got {tuple(emb.shape)}")
    if emb.shape[0] < 1:
        raise RuntimeError("codec produced no frames")
    if not torch.isfinite(emb).all():
        raise RuntimeError("non-finite token embedding")
    return emb, codes, toks


def average_embedding(emb):
    """Mean of one utterance's frame embeddings. ``emb`` is (frames, 256)."""
    if emb.ndim != 2 or emb.shape[0] < 1 or emb.shape[-1] != EMBED_DIM:
        raise RuntimeError(
            f"expected (frames, {EMBED_DIM}) embeddings, got {tuple(getattr(emb, 'shape', ()))}"
        )
    return emb.mean(dim=0)
