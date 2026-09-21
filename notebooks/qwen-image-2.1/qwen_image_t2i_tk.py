"""Text-to-image demo for Qwen-Image 2.1 (OpenVINO GenAI) with a Tkinter window.

Loads locally exported OpenVINO IR models; nothing is downloaded from the network.
Type a prompt, press Generate, and the image is shown in the window (not saved).

Requires a venv with `openvino`, `openvino-genai`, `numpy` and `pillow`:
    python qwen_image_t2i_tk.py
    python qwen_image_t2i_tk.py --model C:\\path\\to\\Qwen-Image-2.1-IR-INT4 --device GPU
"""

from __future__ import annotations

import argparse
import gc
import os
import queue
import random
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image, ImageTk

# Local OpenVINO IR directories (no downloads).
MODEL_DIRS = {
    "FP16": r"C:\openvino\Qwen-Image-2.1-IR-FP16",
    "INT4": r"C:\openvino\Qwen-Image-2.1-IR-INT4",
}
RESOLUTIONS = {
    "64 x 64": (64, 64),
    "256 x 256": (256, 256),
    "512 x 512": (512, 512),
    "1024 x 1024": (1024, 1024),
    "1536 x 1536": (1536, 1536),
    "2048 x 2048 (native)": (2048, 2048),
    "2048 x 1152 (16:9)": (2048, 1152),
    "1152 x 2048 (9:16)": (1152, 2048),
}
DEFAULT_MODEL = "FP16"
DEFAULT_RESOLUTION = "1024 x 1024"
DEFAULT_STEPS = 40
MAX_SEED = 2**31 - 1
PREVIEW_MAX = 768


def available_devices() -> list[str]:
    """OpenVINO devices with AUTO prepended."""
    return list(dict.fromkeys(["AUTO", *ov.Core().available_devices]))


def output_to_image(output: Any) -> Image.Image:
    """Convert an OpenVINO GenAI image output to a PIL image."""
    data = output.data if hasattr(output, "data") else output
    array = np.asarray(data)
    if array.ndim == 4:
        array = array[0]
    return Image.fromarray(array).convert("RGB")


class QwenImageApp:
    def __init__(
        self,
        root: tk.Tk,
        model_dirs: dict[str, str],
        default_model: str,
        devices: list[str],
        default_device: str,
    ) -> None:
        self.root = root
        self.model_dirs = model_dirs
        self.pipe: Any = None
        self.pipe_key: tuple[str, str] | None = None
        self.busy = False
        self.photo: Any = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()

        root.title("Qwen-Image 2.1 - Text to Image (OpenVINO)")
        root.geometry("860x920")

        ttk.Label(root, text="Prompt").pack(anchor="w", padx=8, pady=4)
        self.prompt = tk.Text(root, height=5, wrap="word")
        self.prompt.pack(fill="x", padx=8, pady=4)
        self.prompt.insert(
            "1.0",
            'A cozy coffee shop with a chalkboard sign reading "Qwen Coffee", cinematic lighting',
        )

        model_row = ttk.Frame(root)
        model_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(model_row, text="Model").grid(row=0, column=0, sticky="e", padx=4)
        self.model = tk.StringVar(value=default_model)
        self.model_combo = ttk.Combobox(
            model_row, textvariable=self.model, values=list(model_dirs), width=12, state="readonly"
        )
        self.model_combo.grid(row=0, column=1)
        ttk.Label(model_row, text="Device").grid(row=0, column=2, sticky="e", padx=8)
        self.device = tk.StringVar(value=default_device)
        self.device_combo = ttk.Combobox(
            model_row, textvariable=self.device, values=devices, width=12, state="readonly"
        )
        self.device_combo.grid(row=0, column=3)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        self.device_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        gen_row = ttk.Frame(root)
        gen_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(gen_row, text="Steps").grid(row=0, column=0, sticky="e", padx=4)
        self.steps = tk.IntVar(value=DEFAULT_STEPS)
        ttk.Spinbox(gen_row, from_=1, to=80, width=5, textvariable=self.steps).grid(row=0, column=1)
        ttk.Label(gen_row, text="Resolution").grid(row=0, column=2, sticky="e", padx=8)
        self.resolution = tk.StringVar(value=DEFAULT_RESOLUTION)
        ttk.Combobox(
            gen_row, textvariable=self.resolution, values=list(RESOLUTIONS), width=20, state="readonly"
        ).grid(row=0, column=3)
        ttk.Label(gen_row, text="Seed").grid(row=0, column=4, sticky="e", padx=8)
        self.seed = tk.StringVar(value="42")
        ttk.Entry(gen_row, textvariable=self.seed, width=12).grid(row=0, column=5)
        self.randomize_seed = tk.BooleanVar(value=False)
        ttk.Checkbutton(gen_row, text="Randomize seed", variable=self.randomize_seed).grid(
            row=0, column=6, padx=8
        )

        action = ttk.Frame(root)
        action.pack(fill="x", padx=8, pady=4)
        self.generate_button = ttk.Button(action, text="Generate", command=self.on_generate)
        self.generate_button.pack(side="left")
        self.status = tk.StringVar(value="Ready. The model loads on the first generation.")
        ttk.Label(action, textvariable=self.status).pack(side="left", padx=12)

        self.image_label = ttk.Label(root, anchor="center", text="Generated image appears here")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=4)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_model_change(self, _event: Any = None) -> None:
        if not self.busy:
            self.status.set(f"Model/device changed. Next Generate reloads {self.model.get()}.")

    def on_generate(self) -> None:
        if self.busy:
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            self.status.set("Enter a prompt first.")
            return
        model_key = self.model.get()
        model_dir = self.model_dirs.get(model_key, "")
        if not model_dir or not os.path.isdir(model_dir):
            self.status.set(f"Model directory not found: {model_dir}")
            return

        seed = random.randint(0, MAX_SEED) if self.randomize_seed.get() else self._current_seed()
        try:
            width, height = RESOLUTIONS[self.resolution.get()]
        except KeyError:
            width, height = 1024, 1024

        config = {
            "model_key": model_key,
            "model_dir": model_dir,
            "device": self.device.get(),
            "prompt": prompt,
            "steps": self.steps.get(),
            "width": width,
            "height": height,
            "seed": seed,
        }
        self.busy = True
        self._set_inputs_enabled(False)
        self.status.set("Working...")
        threading.Thread(target=self._worker, args=(config,), daemon=True).start()
        self.root.after(100, self._poll)

    def _current_seed(self) -> int:
        try:
            return int(self.seed.get())
        except ValueError:
            return 42

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.generate_button.state(["!disabled"] if enabled else ["disabled"])
        state = "readonly" if enabled else "disabled"
        self.model_combo.configure(state=state)
        self.device_combo.configure(state=state)

    def _worker(self, config: dict[str, Any]) -> None:
        try:
            self._ensure_pipeline(config["model_dir"], config["device"])
            self.events.put(("status", "Generating..."))
            image = self._generate(config)
            self.events.put(("image", (image, int(config["seed"]), config)))
        except Exception as exc:  # surface model/runtime errors in the status bar
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _ensure_pipeline(self, model_dir: str, device: str) -> None:
        key = (model_dir, device)
        if self.pipe is not None and self.pipe_key == key:
            return
        self.pipe = None
        self.pipe_key = None
        gc.collect()
        self.events.put(
            ("status", f"Loading {os.path.basename(model_dir)} on {device}...")
        )
        self.pipe = ov_genai.Text2ImagePipeline(model_dir, device)
        self.pipe_key = key

    def _generate(self, config: dict[str, Any]) -> Image.Image:
        output = self.pipe.generate(
            config["prompt"],
            guidance_scale=1.0,
            num_inference_steps=int(config["steps"]),
            generator=ov_genai.TorchGenerator(int(config["seed"])),
            height=int(config["height"]),
            width=int(config["width"]),
        )
        return output_to_image(output)

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status.set(str(payload))
                elif kind == "image":
                    image, seed, config = payload
                    self._show(image)
                    self.seed.set(str(seed))  # keep the seed that was actually used
                    self.status.set(
                        f"Done. model={config['model_key']} device={config['device']} "
                        f"seed={seed} steps={config['steps']} "
                        f"{config['width']}x{config['height']}"
                    )
                    self.busy = False
                    self._set_inputs_enabled(True)
                elif kind == "error":
                    self.status.set(str(payload))
                    self.busy = False
                    self._set_inputs_enabled(True)
        except queue.Empty:
            pass

        if self.busy:
            self.root.after(100, self._poll)

    def _show(self, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.photo, text="")

    def _on_close(self) -> None:
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image 2.1 text-to-image Tkinter demo.")
    parser.add_argument(
        "--model",
        default=None,
        help="Extra model directory to use instead of the built-in INT4/FP16 paths.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="OpenVINO device, e.g. AUTO, GPU, GPU.0, CPU, NPU. Defaults to AUTO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dirs = dict(MODEL_DIRS)
    default_model = DEFAULT_MODEL
    if args.model:
        model_dirs["Custom"] = args.model
        default_model = "Custom"

    devices = available_devices()
    default_device = args.device or "AUTO"
    if default_device not in devices:
        devices.append(default_device)

    root = tk.Tk()
    QwenImageApp(root, model_dirs, default_model, devices, default_device)
    root.mainloop()


if __name__ == "__main__":
    main()
