#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import json
import re
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Reference Audio Manager — persistent storage
# ---------------------------------------------------------------------------
def _default_ref_audio_storage_dir() -> Path:
    """Return the repo-local directory for persistent reference audio storage."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".omnivoice" / "reference_audio"


def _default_batch_output_dir() -> Path:
    """Return the repo-local directory for generated batch outputs."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".omnivoice" / "batch_outputs"


def _default_demo_output_dir() -> Path:
    """Return the repo-local directory for single-output demo files."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".omnivoice" / "demo_outputs"


def _safe_path_part(value: str) -> str:
    """Return a filesystem-safe path fragment."""
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in value)
    safe = safe.strip("_")
    return safe or "item"


class ReferenceAudioManager:
    """Manages named reference audio files for the demo.
    
    Each audio is stored with a user-assigned name, accessible across
    the session. Supports upload, list, and delete operations.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the manager with a storage directory.
        
        Args:
            storage_dir: Directory to store audio files. If None, uses a repo-local
                persistent directory under `.omnivoice/reference_audio`.
        """
        if storage_dir is None:
            storage_dir = _default_ref_audio_storage_dir()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.storage_dir / "manifest.json"
        self.gain_file = self.storage_dir / "gains.json"
        self.default_voice_file = self.storage_dir / "default_voice.json"
        self.default_language_file = self.storage_dir / "default_language.json"
        self._load_manifest()
        self._load_gains()
        self._load_default_voice()
        self._load_default_language()

    def _load_gains(self):
        """Load per-audio gain settings from disk."""
        try:
            if self.gain_file.exists():
                with open(self.gain_file, "r") as f:
                    self._gains = json.load(f)
            else:
                self._gains = {}
        except Exception:
            self._gains = {}
        # keep only gains for existing manifest entries
        self._gains = {k: float(v) for k, v in self._gains.items() if k in self._manifest}
        self._save_gains()

    def _save_gains(self):
        with open(self.gain_file, "w") as f:
            json.dump(self._gains, f, indent=2)

    def get_gain(self, name: str) -> float:
        try:
            return float(self._gains.get(name, 0.0))
        except Exception:
            return 0.0

    def set_gain(self, name: str, gain_db: float) -> None:
        if name not in self._manifest:
            raise ValueError(f"Unknown reference audio: {name}")
        try:
            self._gains[name] = float(gain_db)
        except Exception:
            self._gains[name] = 0.0
        self._save_gains()

    def _load_manifest(self):
        """Load or create the manifest file."""
        if self.manifest_file.exists():
            with open(self.manifest_file, "r") as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {}
        # Validate and normalize that all referenced files exist. The manifest
        # may contain legacy entries where the value is a dict or list; try to
        # coerce to a filepath string when possible.
        normalized = {}
        for name, val in list(self._manifest.items()):
            path_str = None
            if isinstance(val, str):
                path_str = val
            elif isinstance(val, dict):
                # common keys that might hold a filepath
                for k in ("path", "filepath", "file", "stored_path", "filename", "file_path"):
                    if k in val and val[k]:
                        path_str = val[k]
                        break
            elif isinstance(val, (list, tuple)) and val:
                # take first string-like entry
                first = val[0]
                if isinstance(first, str):
                    path_str = first
            if path_str and isinstance(path_str, str) and Path(path_str).exists():
                normalized[name] = path_str
        self._manifest = normalized
        self._save_manifest()

    def _save_manifest(self):
        """Save the manifest to disk."""
        with open(self.manifest_file, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def _load_default_voice(self):
        """Load the saved default voice if present."""
        self._default_voice = None
        if self.default_voice_file.exists():
            try:
                with open(self.default_voice_file, "r") as f:
                    data = json.load(f)
                self._default_voice = data.get("default_voice")
            except Exception:
                self._default_voice = None
        if self._default_voice not in self._manifest:
            self._default_voice = None
            self._save_default_voice()

    def _save_default_voice(self):
        """Persist the current default voice selection."""
        with open(self.default_voice_file, "w") as f:
            json.dump({"default_voice": self._default_voice}, f, indent=2)

    def _load_default_language(self):
        """Load the saved default language if present."""
        self._default_language = None
        if self.default_language_file.exists():
            try:
                with open(self.default_language_file, "r") as f:
                    data = json.load(f)
                self._default_language = data.get("default_language")
            except Exception:
                self._default_language = None
        if self._default_language not in _ALL_LANGUAGES:
            self._default_language = None
            self._save_default_language()

    def _save_default_language(self):
        """Persist the current default language selection."""
        with open(self.default_language_file, "w") as f:
            json.dump({"default_language": self._default_language}, f, indent=2)

    def upload_audio(self, audio_path: str, name: str) -> str:
        """Upload and register a reference audio with a user-assigned name.
        
        Args:
            audio_path: Path to the uploaded audio file.
            name: User-friendly name for the audio (e.g., "John's Voice").
        
        Returns:
            The stored audio path.
        
        Raises:
            ValueError: If the name is empty or the audio file doesn't exist.
        """
        if not name or not name.strip():
            raise ValueError("Audio name cannot be empty.")
        if not Path(audio_path).exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        # Sanitize name (remove special chars, keep alphanumeric + spaces)
        safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in name).strip()
        if not safe_name:
            raise ValueError("Audio name must contain at least one alphanumeric character.")

        # Check if name already exists and overwrite
        stored_path = self.storage_dir / f"{safe_name}.wav"
        shutil.copy(audio_path, str(stored_path))
        self._manifest[name] = str(stored_path)
        # Ensure a default gain entry exists
        if name not in getattr(self, "_gains", {}):
            self._gains[name] = 0.0
            self._save_gains()
        self._save_manifest()
        return name

    def get_list(self) -> List[str]:
        """Return list of all registered audio names."""
        return sorted(self._manifest.keys())

    def get_path(self, name: str) -> Optional[str]:
        """Get the file path for a registered audio name."""
        return self._manifest.get(name)

    def delete_audio(self, name: str) -> bool:
        """Delete a registered audio.
        
        Args:
            name: Name of the audio to delete.
        
        Returns:
            True if deletion succeeded, False if name not found.
        """
        if name not in self._manifest:
            return False
        path = Path(self._manifest[name])
        if path.exists():
            path.unlink()
        del self._manifest[name]
        # remove gain entry as well
        if name in getattr(self, "_gains", {}):
            try:
                del self._gains[name]
                self._save_gains()
            except Exception:
                pass
        if self._default_voice == name:
            self._default_voice = None
            self._save_default_voice()
        self._save_manifest()
        return True

    def get_default_voice(self) -> Optional[str]:
        return self._default_voice

    def get_default_language(self) -> Optional[str]:
        return self._default_language

    def set_default_voice(self, name: Optional[str]) -> None:
        if name is not None and name not in self._manifest:
            raise ValueError(f"Unknown reference audio: {name}")
        self._default_voice = name
        self._save_default_voice()

    def set_default_language(self, lang: Optional[str]) -> None:
        if lang is not None and lang not in _ALL_LANGUAGES:
            raise ValueError(f"Unknown language: {lang}")
        self._default_language = lang
        self._save_default_language()

    def clear_all(self):
        """Delete all registered audios (for cleanup)."""
        for path in self._manifest.values():
            Path(path).unlink(missing_ok=True)
        self._manifest.clear()
        self._save_manifest()


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
    audio_manager: Optional[ReferenceAudioManager] = None,
) -> gr.Blocks:

    if audio_manager is None:
        audio_manager = ReferenceAudioManager()

    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_audio_name=None,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            # Resolve reference audio: either from library by name or direct upload
            final_ref_audio = None
            if ref_audio_name and ref_audio_name != "None":
                final_ref_audio = audio_manager.get_path(ref_audio_name)
                if not final_ref_audio:
                    return None, f"Reference audio '{ref_audio_name}' not found in library."
            elif ref_audio:
                final_ref_audio = ref_audio
            
            if not final_ref_audio:
                return None, "Please either select a reference audio from the library or upload one."
            
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=final_ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        demo_output_dir = _default_demo_output_dir()
        demo_output_dir.mkdir(parents=True, exist_ok=True)
        out_path = demo_output_dir / f"omnioutput_{time.time_ns()}.wav"
        waveform = audio[0]
        # Apply global gain if cloning from a library reference
        if mode == "clone" and ref_audio_name:
            try:
                g_db = audio_manager.get_gain(ref_audio_name)
                factor = float(10 ** (float(g_db) / 20.0)) if g_db is not None else 1.0
                waveform = np.asarray(waveform, dtype=np.float32) * factor
            except Exception:
                pass
        sf.write(str(out_path), waveform, sampling_rate)
        return str(out_path), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        # Resolve runtime default language from audio_manager when Auto is used
        runtime_default = audio_manager.get_default_language() or "Auto"
        chosen = value if value is not None and value != "Auto" else runtime_default
        if chosen not in _ALL_LANGUAGES:
            chosen = "Auto"
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=chosen,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Reference Audio Library
            # ==============================================================
            with gr.TabItem("Reference Audio Library"):
                gr.Markdown(
                    """
## Reference Audio Library

Manage your reference audio files for voice cloning. Upload and name them here,
then select them in the Voice Clone tab.
"""
                )

                def _format_ref_audio_list(items: List[str]) -> str:
                    if not items:
                        return "No reference audios saved yet."
                    return "Saved Reference Audios:\n" + "\n".join(
                        f"  • {name}" for name in items
                    )

                def _ref_audio_dropdown_update(default_value: Optional[str] = None):
                    items = audio_manager.get_list()
                    if default_value not in items:
                        default_value = None
                    return gr.update(choices=items, value=default_value)

                initial_ref_items = audio_manager.get_list()
                initial_ref_text = _format_ref_audio_list(initial_ref_items)
                initial_default_voice = audio_manager.get_default_voice()
                if initial_default_voice not in initial_ref_items:
                    initial_default_voice = None
                initial_default_language = audio_manager.get_default_language() or "Auto"
                if initial_default_language not in _ALL_LANGUAGES:
                    initial_default_language = "Auto"

                with gr.Row():
                    with gr.Column(scale=1):
                        ral_upload = gr.Audio(
                            label="Upload Audio / 上传音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        ral_name = gr.Textbox(
                            label="Audio Name / 音频名称",
                            placeholder="e.g., 'John's Voice', 'Female Speaker 1'",
                        )
                        ral_btn = gr.Button("Add to Library / 添加到库", variant="primary")
                        ral_msg = gr.Textbox(label="Message / 消息", interactive=False)
                    with gr.Column(scale=1):
                        ral_list = gr.Textbox(
                            label="Saved Reference Audio List / 已保存的音频列表",
                            lines=10,
                            interactive=False,
                            value=initial_ref_text,
                        )
                        ral_refresh_btn = gr.Button("Refresh / 刷新列表")
                        ral_selected = gr.Dropdown(
                            label="Select to Delete / 选择要删除的音频",
                            choices=initial_ref_items,
                            value=None,
                        )
                        ral_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Gain (dB) / 增益 (dB)", info="Global per-audio gain applied to outputs.")
                        ral_set_gain_btn = gr.Button("Set Gain / 设置增益")
                        ral_delete_btn = gr.Button("Delete Selected / 删除选中", variant="stop")
                        ral_default = gr.Dropdown(
                            label="Default Voice / 默认语音",
                            choices=initial_ref_items,
                            value=initial_default_voice,
                            allow_custom_value=False,
                        )
                        ral_default_btn = gr.Button("Set Default Voice / 设置默认语音")
                        ral_clear_default_btn = gr.Button("Clear Default Voice / 清除默认语音")
                        ral_default_lang = gr.Dropdown(
                            label="Default Language / 默认语种",
                            choices=_ALL_LANGUAGES,
                            value=initial_default_language,
                            allow_custom_value=False,
                        )
                        ral_default_lang_btn = gr.Button("Set Default Language / 设置默认语种")
                        ral_clear_default_lang_btn = gr.Button("Clear Default Language / 清除默认语种")

                def _update_ref_list():
                    """Update the display list."""
                    items = audio_manager.get_list()
                    default_voice = audio_manager.get_default_voice()
                    if default_voice not in items:
                        default_voice = None
                    return (
                        gr.update(value=_format_ref_audio_list(items)),
                        gr.update(choices=items, value=None),
                        gr.update(choices=items, value=default_voice),
                        gr.update(choices=items, value=default_voice),
                        gr.update(choices=items, value=default_voice),
                        gr.update(value=audio_manager.get_gain(default_voice) if default_voice else 0.0),
                    )

                def _add_ref_audio(audio_path, name):
                    """Add a new reference audio to the library."""
                    if not audio_path:
                        return (
                            "Please upload an audio file.",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )
                    if not name or not name.strip():
                        return (
                            "Please enter a name for the audio.",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )
                    try:
                        audio_manager.upload_audio(audio_path, name)
                        items = audio_manager.get_list()
                        default_voice = audio_manager.get_default_voice()
                        if default_voice not in items:
                            default_voice = None
                        return (
                            f"✓ Added '{name}' to library.",
                            gr.update(value=_format_ref_audio_list(items)),
                            gr.update(choices=items, value=None),
                            gr.update(choices=items, value=default_voice),
                            gr.update(choices=items, value=default_voice),
                            gr.update(choices=items, value=default_voice),
                            gr.update(value=audio_manager.get_gain(default_voice) if default_voice else 0.0),
                        )
                    except Exception as e:
                        return (
                            f"Error: {e}",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )

                def _delete_ref_audio(selected_name):
                    """Delete a reference audio from the library."""
                    if not selected_name:
                        return (
                            "Please select an audio to delete.",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )
                    try:
                        success = audio_manager.delete_audio(selected_name)
                        if success:
                            items = audio_manager.get_list()
                            default_voice = audio_manager.get_default_voice()
                            if default_voice not in items:
                                default_voice = None
                            return (
                                f"✓ Deleted '{selected_name}' from library.",
                                gr.update(value=_format_ref_audio_list(items)),
                                gr.update(choices=items, value=None),
                                gr.update(choices=items, value=default_voice),
                                gr.update(choices=items, value=default_voice),
                                gr.update(choices=items, value=default_voice),
                                gr.update(value=audio_manager.get_gain(default_voice) if default_voice else 0.0),
                            )
                        else:
                            return (
                                f"Audio '{selected_name}' not found.",
                                gr.update(),
                                gr.update(),
                                gr.update(),
                            )
                    except Exception as e:
                        return (
                            f"Error: {e}",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )

                def _set_default_voice(selected_name):
                    if not selected_name:
                        return (
                            "Please select a reference voice.",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )
                    try:
                        audio_manager.set_default_voice(selected_name)
                        items = audio_manager.get_list()
                        return (
                            f"✓ Default voice set to '{selected_name}'.",
                            gr.update(value=_format_ref_audio_list(items)),
                            gr.update(choices=items, value=None),
                            gr.update(choices=items, value=selected_name),
                            gr.update(choices=items, value=selected_name),
                            gr.update(choices=items, value=selected_name),
                            gr.update(value=audio_manager.get_gain(selected_name) if selected_name else 0.0),
                        )
                    except Exception as e:
                        return (
                            f"Error: {e}",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )

                def _clear_default_voice():
                    audio_manager.set_default_voice(None)
                    items = audio_manager.get_list()
                    return (
                        "✓ Default voice cleared.",
                        gr.update(value=_format_ref_audio_list(items)),
                        gr.update(choices=items, value=None),
                        gr.update(choices=items, value=None),
                        gr.update(choices=items, value=None),
                        gr.update(choices=items, value=None),
                        gr.update(value=0.0),
                    )

                def _set_default_language(selected_lang):
                    if not selected_lang:
                        return (
                            "Please select a language.",
                            gr.update(),
                        )
                    try:
                        audio_manager.set_default_language(selected_lang)
                        return (
                            f"✓ Default language set to '{selected_lang}'.",
                            gr.update(value=selected_lang),
                        )
                    except Exception as e:
                        return (
                            f"Error: {e}",
                            gr.update(),
                        )

                def _clear_default_language():
                    audio_manager.set_default_language(None)
                    return (
                        "✓ Default language cleared.",
                        gr.update(value="Auto"),
                    )

                def _on_ref_selected(name: str):
                    if not name:
                        return gr.update(value=0.0)
                    return gr.update(value=audio_manager.get_gain(name))

                def _set_ref_gain(selected_name, gain_value):
                    if not selected_name:
                        return "Please select an audio to set gain for.", gr.update(value=0.0)
                    try:
                        audio_manager.set_gain(selected_name, float(gain_value))
                        return f"✓ Gain for '{selected_name}' set to {gain_value} dB.", gr.update(value=float(gain_value))
                    except Exception as e:
                        return f"Error: {e}", gr.update()

            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        gr.Markdown("### Reference Audio")
                        vc_ref_audio_name = gr.Dropdown(
                            label="Select from Library / 从库中选择",
                            choices=initial_ref_items,
                            value=initial_default_voice,
                            allow_custom_value=False,
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Or upload a new one (library will be ignored if provided):"
                            "</span>"
                        )
                        vc_ref_audio = gr.Audio(
                            label="Or Upload New Reference Audio / 或上传新音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional)" " / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="filepath",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(
                    text, lang, ref_aud_name, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_audio_name=ref_aud_name,
                        ref_text=ref_text or None,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio_name,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

                ral_btn.click(
                    _add_ref_audio,
                    inputs=[ral_upload, ral_name],
                    outputs=[ral_msg, ral_list, ral_selected, ral_default, vc_ref_audio_name, ral_gain],
                )
                ral_refresh_btn.click(
                    _update_ref_list,
                    outputs=[ral_list, ral_selected, ral_default, vc_ref_audio_name, ral_gain],
                )
                ral_delete_btn.click(
                    _delete_ref_audio,
                    inputs=[ral_selected],
                    outputs=[ral_msg, ral_list, ral_selected, ral_default, vc_ref_audio_name, ral_gain],
                )
                ral_default_btn.click(
                    _set_default_voice,
                    inputs=[ral_default],
                    outputs=[ral_msg, ral_list, ral_selected, ral_default, vc_ref_audio_name, ral_gain],
                )
                ral_clear_default_btn.click(
                    _clear_default_voice,
                    outputs=[ral_msg, ral_list, ral_selected, ral_default, vc_ref_audio_name, ral_gain],
                )
                ral_selected.change(_on_ref_selected, inputs=[ral_selected], outputs=[ral_gain])
                ral_set_gain_btn.click(_set_ref_gain, inputs=[ral_selected, ral_gain], outputs=[ral_msg, ral_gain])
                

            # ==============================================================
            # Batch Generate
            # ==============================================================
            with gr.TabItem("Batch Generate"):
                gr.Markdown(
                    """
## Batch Generate

Add or remove rows, then assign each text its own voice from the library
when available. Rows without a selected voice can still use voice design
or auto voice.
"""
                )

                batch_count = gr.State(1)
                BATCH_MAX_ROWS = 8
                batch_row_containers = []
                batch_textboxes = []
                batch_voice_dropdowns = []
                batch_ref_textboxes = []
                batch_lang_dropdowns = []
                batch_instruct_textboxes = []

                def _batch_row_visibility(count: int):
                    return [gr.update(visible=i < count) for i in range(BATCH_MAX_ROWS)]

                def _batch_status_text(count: int):
                    return (
                        f"Showing {count} batch item(s). "
                        f"Use the voice library dropdown per row when you want cloning."
                    )

                def _batch_library_choices_update():
                    items = audio_manager.get_list()
                    default_voice = audio_manager.get_default_voice()
                    if default_voice not in items:
                        default_voice = None
                    return [gr.update(choices=items, value=default_voice) for _ in batch_voice_dropdowns]

                def _plain_value(v):
                    return getattr(v, "value", v)

                with gr.Row():
                    batch_add_btn = gr.Button("Add Item / 添加一行", variant="primary")
                    batch_remove_btn = gr.Button("Remove Item / 删除一行", variant="secondary")
                    batch_refresh_btn = gr.Button("Refresh Voices / 刷新语音库", variant="secondary")

                batch_status = gr.Textbox(
                    label="Batch Status / 批量状态",
                    value=_batch_status_text(1),
                    interactive=False,
                )

                (
                    batch_ns,
                    batch_gs,
                    batch_dn,
                    batch_sp,
                    batch_du,
                    batch_pp,
                    batch_po,
                ) = _gen_settings()
                batch_output_files = gr.Audio(
                    label="Merged Batch Audio / 合并结果",
                    type="filepath",
                )

                for i in range(BATCH_MAX_ROWS):
                    with gr.Row(visible=(i == 0)) as batch_row:
                        with gr.Column(scale=2):
                            text_i = gr.Textbox(
                                label=f"Text {i + 1} / 文本 {i + 1}",
                                lines=3,
                                placeholder="Enter the text you want to synthesize...",
                            )
                        with gr.Column(scale=1):
                            voice_i = gr.Dropdown(
                                label=f"Voice {i + 1} / 语音 {i + 1}",
                                choices=initial_ref_items,
                                value=initial_default_voice,
                                allow_custom_value=False,
                                info="Select a named voice from the library if you want voice cloning.",
                            )
                            ref_text_i = gr.Textbox(
                                label=f"Reference Text {i + 1} (optional)",
                                lines=2,
                                placeholder="Transcript for the selected reference audio.",
                            )
                            lang_i = _lang_dropdown(f"Language {i + 1} (optional)", value="Auto")
                            instruct_i = gr.Textbox(
                                label=f"Instruct {i + 1} (optional)",
                                lines=2,
                                placeholder="Style description for voice design if no voice is selected.",
                            )

                    batch_row_containers.append(batch_row)
                    batch_textboxes.append(text_i)
                    batch_voice_dropdowns.append(voice_i)
                    batch_ref_textboxes.append(ref_text_i)
                    batch_lang_dropdowns.append(lang_i)
                    batch_instruct_textboxes.append(instruct_i)

                    

                # Keep batch dropdowns in sync when library changes
                ral_btn.click(lambda: _batch_library_choices_update(), outputs=batch_voice_dropdowns)
                ral_refresh_btn.click(lambda: _batch_library_choices_update(), outputs=batch_voice_dropdowns)
                ral_delete_btn.click(lambda: _batch_library_choices_update(), outputs=batch_voice_dropdowns)
                ral_default_btn.click(lambda: _batch_library_choices_update(), outputs=batch_voice_dropdowns)
                ral_clear_default_btn.click(lambda: _batch_library_choices_update(), outputs=batch_voice_dropdowns)

                def _batch_add(current_count: int):
                    new_count = max(1, min(BATCH_MAX_ROWS, current_count + 1))
                    return [new_count, *(_batch_row_visibility(new_count)), _batch_status_text(new_count)]

                def _batch_remove(current_count: int):
                    new_count = max(1, min(BATCH_MAX_ROWS, current_count - 1))
                    return [new_count, *(_batch_row_visibility(new_count)), _batch_status_text(new_count)]

                def _batch_generate(
                    current_count,
                    num_step,
                    guidance_scale,
                    denoise,
                    speed_setting,
                    duration_setting,
                    preprocess_prompt,
                    postprocess_output,
                    *row_values,
                ):
                    current_count = _plain_value(current_count)
                    num_step = _plain_value(num_step)
                    guidance_scale = _plain_value(guidance_scale)
                    denoise = _plain_value(denoise)
                    speed_setting = _plain_value(speed_setting)
                    duration_setting = _plain_value(duration_setting)
                    preprocess_prompt = _plain_value(preprocess_prompt)
                    postprocess_output = _plain_value(postprocess_output)

                    rows = []
                    step = 5
                    for i in range(BATCH_MAX_ROWS):
                        offset = i * step
                        text = row_values[offset]
                        voice_name = row_values[offset + 1]
                        ref_text = row_values[offset + 2]
                        lang = row_values[offset + 3]
                        instruct = row_values[offset + 4]
                        if i >= int(current_count or 1):
                            continue
                        if not text or not str(text).strip():
                            continue
                        rows.append(
                            {
                                "index": i,
                                "text": str(text).strip(),
                                "voice_name": voice_name if voice_name else None,
                                "ref_text": ref_text.strip() if isinstance(ref_text, str) and ref_text.strip() else None,
                                "lang": lang if lang and lang != "Auto" else None,
                                "instruct": instruct.strip() if isinstance(instruct, str) and instruct.strip() else None,
                            }
                        )

                    if not rows:
                        return [], "No batch items to generate."

                    gen_config = OmniVoiceGenerationConfig(
                        num_step=int(num_step or 32),
                        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
                        denoise=bool(denoise) if denoise is not None else True,
                        preprocess_prompt=bool(preprocess_prompt),
                        postprocess_output=bool(postprocess_output),
                    )
                    duration = float(duration_setting) if duration_setting is not None and float(duration_setting) > 0 else None
                    speed = float(speed_setting) if speed_setting is not None and float(speed_setting) != 1.0 else None
                    batch_output_dir = _default_batch_output_dir()
                    batch_output_dir.mkdir(parents=True, exist_ok=True)

                    def _merge_audios(audio_arrays: List[np.ndarray]) -> np.ndarray:
                        if not audio_arrays:
                            return np.zeros(0, dtype=np.float32)
                        separator = np.zeros(int(model.sampling_rate * 0.35), dtype=np.float32)
                        merged_parts = []
                        for idx, audio_array in enumerate(audio_arrays):
                            audio_np = np.asarray(audio_array, dtype=np.float32).reshape(-1)
                            if audio_np.size == 0:
                                continue
                            merged_parts.append(audio_np)
                            if idx != len(audio_arrays) - 1:
                                merged_parts.append(separator)
                        if not merged_parts:
                            return np.zeros(0, dtype=np.float32)
                        return np.concatenate(merged_parts)

                    def _save_audio(audio_array, row_index: int, voice_name: Optional[str]):
                        out_path = batch_output_dir / f"omnioutput_{time.time_ns()}.wav"
                        sf.write(str(out_path), audio_array, model.sampling_rate)
                        return str(out_path)

                    row_audios = []

                    try:
                        for r in sorted(rows, key=lambda x: x["index"]):
                            gen_kwargs = dict(
                                text=r["text"],
                                language=r["lang"],
                                instruct=r["instruct"],
                                duration=duration,
                                speed=speed,
                                generation_config=gen_config,
                            )

                            if r["voice_name"]:
                                ref_path = audio_manager.get_path(r["voice_name"])
                                if not ref_path:
                                    raise ValueError(
                                        f"Reference voice '{r['voice_name']}' not found in the library."
                                    )
                                gen_kwargs["voice_clone_prompt"] = model.create_voice_clone_prompt(
                                    ref_audio=ref_path,
                                    ref_text=r["ref_text"],
                                    preprocess_prompt=bool(preprocess_prompt),
                                )

                            audio = model.generate(**gen_kwargs)
                            waveform = audio[0]
                            # Apply per-audio global gain if voice_name provided
                            if r.get("voice_name"):
                                try:
                                    g_db = audio_manager.get_gain(r.get("voice_name"))
                                    factor = float(10 ** (float(g_db) / 20.0)) if g_db is not None else 1.0
                                except Exception:
                                    factor = 1.0
                                waveform = np.asarray(waveform, dtype=np.float32) * factor
                            row_audios.append(waveform)

                    except Exception as e:
                        return None, f"Error: {type(e).__name__}: {e}"

                    merged_audio = _merge_audios(row_audios)
                    if merged_audio.size == 0:
                        return None, "No valid audio was generated."

                    merged_path = _save_audio(merged_audio, 0, None)

                    summary = (
                        f"Generated 1 merged file in {batch_output_dir}. "
                        f"Saved outputs persist across restarts."
                    )
                    return merged_path, summary

                batch_add_btn.click(
                    _batch_add,
                    inputs=[batch_count],
                    outputs=[batch_count, *batch_row_containers, batch_status],
                )
                # Ensure newly-added rows pick up the current library choices/default
                batch_add_btn.click(lambda: _batch_library_choices_update(), outputs=batch_voice_dropdowns)
                batch_remove_btn.click(
                    _batch_remove,
                    inputs=[batch_count],
                    outputs=[batch_count, *batch_row_containers, batch_status],
                )
                batch_remove_btn.click(lambda: _batch_library_choices_update(), outputs=batch_voice_dropdowns)
                batch_refresh_btn.click(
                    lambda: _batch_library_choices_update(),
                    outputs=batch_voice_dropdowns,
                )

                batch_btn = gr.Button("Generate Batch / 批量生成", variant="primary")
                batch_inputs = []
                for i in range(BATCH_MAX_ROWS):
                    batch_inputs.extend(
                        [
                            batch_textboxes[i],
                            batch_voice_dropdowns[i],
                            batch_ref_textboxes[i],
                            batch_lang_dropdowns[i],
                            batch_instruct_textboxes[i],
                        ]
                    )
                batch_btn.click(
                    _batch_generate,
                    inputs=[
                        batch_count,
                        batch_ns,
                        batch_gs,
                        batch_dn,
                        batch_sp,
                        batch_du,
                        batch_pp,
                        batch_po,
                        *batch_inputs,
                    ],
                    outputs=[batch_output_files, batch_status],
                )

            # ==============================================================
            # Script Parser
            # ==============================================================
            with gr.TabItem("Script Parser"):
                gr.Markdown(
                    """
## Script Parser

Paste a script where each spoken line may start with a speaker tag:
`[[speaker name]]: text`.
Each parsed line becomes an editable row. Speaker names default to the
saved default voice when available. Lines without a speaker tag are
kept and assigned the default voice as well.
"""
                )

                script_count = gr.State(0)
                SCRIPT_MAX_LINES = 32
                SCRIPT_MAX_SPEAKERS = 16
                script_row_containers = []
                script_textboxes = []
                script_speaker_boxes = []
                script_voice_dropdowns = []
                script_lang_dropdowns = []
                script_speaker_map_rows = []
                script_speaker_map_names = []
                script_speaker_map_voice_dropdowns = []

                def _script_row_visibility(count: int):
                    return [gr.update(visible=i < count) for i in range(SCRIPT_MAX_LINES)]

                def _parse_script(text: str):
                    text = str(text) if text is not None else ""
                    lines = [l.strip() for l in str(text).splitlines() if l.strip()] if text.strip() else []
                    parsed = []
                    for l in lines:
                        # First extract a speaker tag anywhere in the line (do not remove yet)
                        sm = re.search(r"\[\[\s*(.*?)\s*\]\]", l)
                        speaker = sm.group(1).strip() if sm else None
                        # Now remove any [[...]] tags and optional following ':' from the line
                        content = re.sub(r"\[\[.*?\]\]\s*:?", "", l).strip()
                        parsed.append((speaker, content))
                        if len(parsed) >= SCRIPT_MAX_LINES:
                            break

                    count = len(parsed)
                    vis = _script_row_visibility(count)
                    # Build cleaned text (remove any [[speaker]] tags)
                    cleaned_lines = [content for (_spk, content) in parsed]
                    cleaned_text = "\n".join(cleaned_lines)
                    # Prepare per-row updates: text, hidden speaker
                    updates = []
                    items = audio_manager.get_list()
                    default_voice = audio_manager.get_default_voice() if audio_manager.get_default_voice() in items else None
                    default_lang = audio_manager.get_default_language() or "Auto"
                    for i in range(SCRIPT_MAX_LINES):
                        if i < count:
                            spk, txt = parsed[i]
                            label_text = f"Line {i+1} — {spk}" if spk else f"Line {i+1}"
                            updates.append(gr.update(value=txt, label=label_text))
                            # hidden speaker value
                            updates.append(gr.update(value=spk or ""))
                        else:
                            label_text = f"Line {i+1}"
                            updates.append(gr.update(value="", label=label_text))
                            updates.append(gr.update(value=""))

                    # Prepare speaker mapping updates (names, voices, gains)
                    unique_speakers = []
                    for spk, _ct in parsed:
                        if spk and spk not in unique_speakers:
                            unique_speakers.append(spk)

                    mapping_row_updates = []
                    mapping_name_updates = []
                    mapping_voice_updates = []
                    for i in range(SCRIPT_MAX_SPEAKERS):
                        if i < len(unique_speakers):
                            mapping_row_updates.append(gr.update(visible=True))
                            mapping_name_updates.append(gr.update(value=unique_speakers[i], visible=True))
                            mapping_voice_updates.append(gr.update(choices=items, value=default_voice, visible=True))
                        else:
                            mapping_row_updates.append(gr.update(visible=False))
                            mapping_name_updates.append(gr.update(value="", visible=False))
                            mapping_voice_updates.append(gr.update(choices=items, value=None, visible=False))

                    return [cleaned_text, count, *mapping_row_updates, *mapping_name_updates, *mapping_voice_updates, *vis, *updates]

                with gr.Row():
                    script_input = gr.TextArea(label="Paste Script / 粘贴脚本", lines=8)
                    script_file = gr.File(label="Load .txt File", file_count="single", file_types=[".txt"], type="filepath")
                    script_parse_btn = gr.Button("Parse Script / 解析脚本", variant="primary")

                with gr.Accordion("Speaker Mappings (per unique speaker)", open=True):
                    for i in range(SCRIPT_MAX_SPEAKERS):
                        with gr.Row(visible=False) as map_row:
                            m_name = gr.Textbox(label=f"Speaker {i+1}", interactive=False)
                            m_voice = gr.Dropdown(label=f"Voice for Speaker {i+1}", choices=initial_ref_items, value=initial_default_voice, allow_custom_value=False)
                        script_speaker_map_rows.append(map_row)
                        script_speaker_map_names.append(m_name)
                        script_speaker_map_voice_dropdowns.append(m_voice)
                        

                with gr.Row():
                    script_parse_msg = gr.Textbox(label="Message", interactive=False)

                # Preview player for generated script audio
                with gr.Row():
                    script_preview_audio = gr.Audio(label="Preview Audio / 预览音频", type="filepath")

                for i in range(SCRIPT_MAX_LINES):
                    with gr.Row(visible=False) as script_row:
                        with gr.Column(scale=3):
                            s_text = gr.Textbox(label=f"Line {i+1}", lines=2)
                        # hidden speaker value (not shown to user) used for mapping
                        s_speaker = gr.Textbox(visible=False)

                    script_row_containers.append(script_row)
                    script_textboxes.append(s_text)
                    script_speaker_boxes.append(s_speaker)

                # Wire parse button to populate rows (include script_input as first output)
                outputs = [script_input, script_count]
                # include speaker mapping outputs (rows, then names then voices)
                outputs.extend(script_speaker_map_rows)
                outputs.extend(script_speaker_map_names)
                outputs.extend(script_speaker_map_voice_dropdowns)
                # then row containers
                outputs.extend(script_row_containers)
                # For each row: text and hidden speaker
                for i in range(SCRIPT_MAX_LINES):
                    outputs.extend([script_textboxes[i], script_speaker_boxes[i]])

                def _load_and_parse(file_path: str):
                    def _resolve_path(fp):
                        if not fp:
                            return None
                        if isinstance(fp, str):
                            return fp
                        if isinstance(fp, dict):
                            # common keys gradio may provide
                            for k in ("tmp_path", "tempfile", "file_path", "filepath", "name", "filename"):
                                if k in fp and fp[k]:
                                    return fp[k]
                            # sometimes file object is nested
                            return fp.get("name") or fp.get("filename")
                        if isinstance(fp, (list, tuple)) and fp:
                            return _resolve_path(fp[0])
                        return None

                    path = _resolve_path(file_path)
                    if not path:
                        return _parse_script("")
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                    except Exception:
                        try:
                            with open(path, "r", encoding="latin-1") as f:
                                content = f.read()
                        except Exception:
                            return _parse_script("")
                    return _parse_script(content)

                script_parse_btn.click(_parse_script, inputs=[script_input], outputs=outputs)
                script_file.change(_load_and_parse, inputs=[script_file], outputs=outputs)

                def _script_generate(current_count, num_step, guidance_scale, denoise, speed_setting, duration_setting, preprocess_prompt, postprocess_output, *row_values):
                    # Reuse batch-style generator logic
                    current_count = _plain_value(current_count)
                    num_step = _plain_value(num_step)
                    guidance_scale = _plain_value(guidance_scale)
                    denoise = _plain_value(denoise)
                    speed_setting = _plain_value(speed_setting)
                    duration_setting = _plain_value(duration_setting)
                    preprocess_prompt = _plain_value(preprocess_prompt)
                    postprocess_output = _plain_value(postprocess_output)

                    # First part of row_values contains speaker mapping (names then voices)
                    mapping_count = SCRIPT_MAX_SPEAKERS * 2
                    mapping_vals = list(row_values[:mapping_count])
                    # mapping names are first half, voices second half
                    mapping_names = [mapping_vals[i] for i in range(0, SCRIPT_MAX_SPEAKERS)]
                    mapping_voices = [mapping_vals[i] for i in range(SCRIPT_MAX_SPEAKERS, SCRIPT_MAX_SPEAKERS * 2)]
                    speaker_to_voice = {}
                    for n, v in zip(mapping_names, mapping_voices):
                        if n and str(n).strip():
                            speaker_to_voice[str(n).strip()] = v if v else None

                    rows = []
                    step = 2
                    # remaining values correspond to per-row fields (text, hidden speaker)
                    row_vals_offset = mapping_count
                    for i in range(SCRIPT_MAX_LINES):
                        offset = row_vals_offset + i * step
                        text = row_values[offset]
                        speaker = row_values[offset + 1]
                        if i >= int(current_count or 0):
                            continue
                        if not text or not str(text).strip():
                            continue
                        # Resolve voice from mapping for speaker
                        resolved_voice = None
                        if speaker and str(speaker).strip():
                            resolved_voice = speaker_to_voice.get(str(speaker).strip())
                        rows.append({
                            "index": i,
                            "text": str(text).strip(),
                            "speaker": speaker if speaker else None,
                            "voice_name": resolved_voice if resolved_voice else None,
                            "lang": audio_manager.get_default_language() if audio_manager.get_default_language() and audio_manager.get_default_language() != "Auto" else None,
                        })

                    if not rows:
                        return None, "No script lines to generate."

                    gen_config = OmniVoiceGenerationConfig(
                        num_step=int(num_step or 32),
                        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
                        denoise=bool(denoise) if denoise is not None else True,
                        preprocess_prompt=bool(preprocess_prompt),
                        postprocess_output=bool(postprocess_output),
                    )
                    duration = float(duration_setting) if duration_setting is not None and float(duration_setting) > 0 else None
                    speed = float(speed_setting) if speed_setting is not None and float(speed_setting) != 1.0 else None
                    batch_output_dir = _default_batch_output_dir()
                    batch_output_dir.mkdir(parents=True, exist_ok=True)

                    def _merge_audios(audio_arrays: List[np.ndarray]) -> np.ndarray:
                        if not audio_arrays:
                            return np.zeros(0, dtype=np.float32)
                        separator = np.zeros(int(model.sampling_rate * 0.35), dtype=np.float32)
                        merged_parts = []
                        for idx, audio_array in enumerate(audio_arrays):
                            audio_np = np.asarray(audio_array, dtype=np.float32).reshape(-1)
                            if audio_np.size == 0:
                                continue
                            merged_parts.append(audio_np)
                            if idx != len(audio_arrays) - 1:
                                merged_parts.append(separator)
                        if not merged_parts:
                            return np.zeros(0, dtype=np.float32)
                        return np.concatenate(merged_parts)

                    row_audios = []
                    try:
                        for r in sorted(rows, key=lambda x: x["index"]):
                            gen_kwargs = dict(
                                text=r["text"],
                                language=r["lang"],
                                duration=duration,
                                speed=speed,
                                generation_config=gen_config,
                            )

                            if r["voice_name"]:
                                ref_path = audio_manager.get_path(r["voice_name"])
                                if not ref_path:
                                    raise ValueError(f"Reference voice '{r['voice_name']}' not found in the library.")
                                gen_kwargs["voice_clone_prompt"] = model.create_voice_clone_prompt(
                                    ref_audio=ref_path,
                                    ref_text=None,
                                    preprocess_prompt=bool(preprocess_prompt),
                                )

                            audio = model.generate(**gen_kwargs)
                            waveform = audio[0]
                            # Apply global per-audio gain if this row used a library voice
                            voice_used = r.get("voice_name")
                            if voice_used:
                                try:
                                    g_db = audio_manager.get_gain(voice_used)
                                    factor = float(10 ** (float(g_db) / 20.0)) if g_db is not None else 1.0
                                except Exception:
                                    factor = 1.0
                                waveform = np.asarray(waveform, dtype=np.float32) * factor
                            row_audios.append(waveform)

                    except Exception as e:
                        return None, f"Error: {type(e).__name__}: {e}"

                    merged_audio = _merge_audios(row_audios)
                    if merged_audio.size == 0:
                        return None, "No valid audio was generated."

                    merged_path = batch_output_dir / f"omnioutput_{time.time_ns()}.wav"
                    sf.write(str(merged_path), merged_audio, model.sampling_rate)

                    summary = f"Generated merged file: {merged_path}"
                    return str(merged_path), summary

                script_generate_btn = gr.Button("Generate Script Audio / 生成脚本音频", variant="primary")
                # Build inputs list: count + shared generation settings + mapping inputs + per-row fields
                script_inputs = [
                    script_count,
                    batch_ns,
                    batch_gs,
                    batch_dn,
                    batch_sp,
                    batch_du,
                    batch_pp,
                    batch_po,
                ]
                # mapping names then mapping voices
                script_inputs.extend(script_speaker_map_names)
                script_inputs.extend(script_speaker_map_voice_dropdowns)
                for i in range(SCRIPT_MAX_LINES):
                    script_inputs.extend([script_textboxes[i], script_speaker_boxes[i]])

                script_generate_btn.click(_script_generate, inputs=script_inputs, outputs=[script_preview_audio, script_parse_msg])


            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="filepath",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

                # Wire default language controls to update language widgets across tabs
                try:
                    ral_default_lang_btn.click(
                        _set_default_language,
                        inputs=[ral_default_lang],
                        outputs=[ral_msg, ral_default_lang],
                    )
                    ral_clear_default_lang_btn.click(
                        _clear_default_language,
                        outputs=[ral_msg, ral_default_lang],
                    )

                    # Propagate the saved default language to each language dropdown
                    ral_default_lang_btn.click(
                        lambda: gr.update(value=audio_manager.get_default_language() or "Auto"),
                        outputs=[vc_lang],
                    )
                    ral_default_lang_btn.click(
                        lambda: [gr.update(value=audio_manager.get_default_language() or "Auto") for _ in batch_lang_dropdowns],
                        outputs=batch_lang_dropdowns,
                    )
                    ral_default_lang_btn.click(
                        lambda: gr.update(value=audio_manager.get_default_language() or "Auto"),
                        outputs=[vd_lang],
                    )

                    ral_clear_default_lang_btn.click(
                        lambda: gr.update(value="Auto"),
                        outputs=[vc_lang],
                    )
                    ral_clear_default_lang_btn.click(
                        lambda: [gr.update(value="Auto") for _ in batch_lang_dropdowns],
                        outputs=batch_lang_dropdowns,
                    )
                    ral_clear_default_lang_btn.click(
                        lambda: gr.update(value="Auto"),
                        outputs=[vd_lang],
                    )
                except Exception:
                    # Defensive: if wiring fails (components not present), skip propagation
                    pass

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    audio_manager = ReferenceAudioManager()
    demo = build_demo(model, checkpoint, audio_manager=audio_manager)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
