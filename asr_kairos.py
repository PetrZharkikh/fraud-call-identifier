from __future__ import annotations

import sys
from typing import Optional, Tuple

_asr = None
_asr_key: Optional[Tuple[Optional[str], str, bool]] = None


def _require_py310() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "kairos-asr требует Python 3.10+. Запустите: python3 predict.py ... "
            "или python3.10 ..."
        )


def get_kairos(
    device: str = "auto",
    model_path: Optional[str] = None,
    force_download: bool = False,
):
    global _asr, _asr_key
    _require_py310()
    key = (model_path, device, force_download)
    if _asr is None or _asr_key != key:
        from kairos_asr import KairosASR

        _asr = KairosASR(
            model_path=model_path,
            device=device,
            force_download=force_download,
        )
        _asr_key = key
    return _asr


def transcribe_wav(
    wav_path: str,
    *,
    device: str = "auto",
    model_path: Optional[str] = None,
    force_download: bool = False,
    pause_threshold: float = 2.0,
) -> str:
    asr = get_kairos(
        device=device,
        model_path=model_path,
        force_download=force_download,
    )
    result = asr.transcribe(wav_file=wav_path, pause_threshold=pause_threshold)
    return (result.full_text or "").strip()
