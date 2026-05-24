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
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
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
        self._load_manifest()

    def _load_manifest(self):
        """Load or create the manifest file."""
        if self.manifest_file.exists():
            with open(self.manifest_file, "r") as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {}
        # Validate that all referenced files exist
        self._manifest = {
            name: path
            for name, path in self._manifest.items()
            if Path(path).exists()
        }
        self._save_manifest()

    def _save_manifest(self):
        """Save the manifest to disk."""
        with open(self.manifest_file, "w") as f:
            json.dump(self._manifest, f, indent=2)

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
        self._save_manifest()
        return True

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

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

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
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
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

                def _ref_audio_dropdown_update():
                    items = audio_manager.get_list()
                    return gr.update(choices=items, value=None)

                initial_ref_items = audio_manager.get_list()
                initial_ref_text = _format_ref_audio_list(initial_ref_items)

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
                        ral_delete_btn = gr.Button("Delete Selected / 删除选中", variant="stop")

                def _update_ref_list():
                    """Update the display list."""
                    items = audio_manager.get_list()
                    return (
                        gr.update(value=_format_ref_audio_list(items)),
                        gr.update(choices=items, value=None),
                        gr.update(choices=items, value=None),
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
                        return (
                            f"✓ Added '{name}' to library.",
                            gr.update(value=_format_ref_audio_list(items)),
                            gr.update(choices=items, value=None),
                            gr.update(choices=items, value=None),
                        )
                    except Exception as e:
                        return (
                            f"Error: {e}",
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
                            return (
                                f"✓ Deleted '{selected_name}' from library.",
                                gr.update(value=_format_ref_audio_list(items)),
                                gr.update(choices=items, value=None),
                                gr.update(choices=items, value=None),
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
                        )

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
                            value=None,
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
                            type="numpy",
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
                    outputs=[ral_msg, ral_list, ral_selected, vc_ref_audio_name],
                )
                ral_refresh_btn.click(
                    _update_ref_list,
                    outputs=[ral_list, ral_selected, vc_ref_audio_name],
                )
                ral_delete_btn.click(
                    _delete_ref_audio,
                    inputs=[ral_selected],
                    outputs=[ral_msg, ral_list, ral_selected, vc_ref_audio_name],
                )

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
                            type="numpy",
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
