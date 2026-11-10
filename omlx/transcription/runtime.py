# SPDX-License-Identifier: Apache-2.0
"""Runtime state belongs to the selected oMLX base path, never site-packages."""
from pathlib import Path


def runtime_directory(name):
    from omlx.settings import get_settings
    directory = Path(get_settings().base_path) / 'transcription' / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_nemotron_model(model):
    try:
        from mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe import Model
    except ImportError:
        return False
    return isinstance(model, Model)
