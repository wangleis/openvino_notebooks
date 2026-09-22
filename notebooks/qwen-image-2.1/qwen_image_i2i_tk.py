"""Image-to-image demo for Qwen-Image 2.1 (OpenVINO GenAI) with a Tkinter window.

Loads locally exported OpenVINO IR models; nothing is downloaded from the network.
Add up to ten input images, stitch them into a single condition image (Qwen-Image-Edit
style), type an editing instruction, press Generate, and the result is shown in the
window (not saved unless you press Save).

Requires a venv with `openvino`, `openvino-genai`, `numpy` and `pillow`:
    python qwen_image_i2i_tk.py
    python qwen_image_i2i_tk.py --model C:\\path\\to\\Qwen-Image-2.1-IR-INT4 --device GPU
    python qwen_image_i2i_tk.py --images a.png b.png --prompt "Combine image 1 and image 2."
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import queue
import random
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image, ImageOps, ImageTk

# Local OpenVINO IR directories (no downloads).
MODEL_DIRS = {
    "FP16": r"C:\openvino\Qwen-Image-2.1-IR-FP16",
    "INT4": r"C:\openvino\Qwen-Image-2.1-IR-INT4",
}
DEFAULT_MODEL = "FP16"
MAX_INPUT_IMAGES = 10
DEFAULT_STEPS = 40
MAX_SEED = 2**31 - 1
PREVIEW_MAX = 768
TARGET_CONDITION_PIXELS = 1024 * 1024
DEFAULT_PROMPT = (
    "Combine the subject from image 1 with the scene from image 2 into one coherent, "
    "photorealistic image. Preserve each subject's identity and match the lighting and "
    "perspective."
)


def available_devices() -> list[str]:
    """Devices reported by the OpenVINO runtime."""
    return list(ov.Core().available_devices)


def resolve_device(requested: str, available: list[str]) -> str:
    """Resolve a requested device to a concrete available device, falling back to CPU.

    openvino_genai exposes no API to read back the real execution device, so we resolve
    it ourselves and pass an explicit device to the pipeline. The returned value is the
    device the pipeline actually runs on (and what the UI reports).
    """
    if requested == "AUTO":
        # AUTO default priority is dGPU -> iGPU -> CPU (NPU is excluded by default).
        gpus = [device for device in available if device.startswith("GPU")]
        return gpus[0] if gpus else "CPU"
    if requested in available:
        return requested
    for device in available:
        if device.startswith(requested + "."):
            return device
    return "CPU"


def output_to_image(output: Any) -> Image.Image:
    """Convert an OpenVINO GenAI image output to a PIL image."""
    data = output.data if hasattr(output, "data") else output
    array = np.asarray(data)
    if array.ndim == 4:
        array = array[0]
    return Image.fromarray(array).convert("RGB")


def image_to_tensor(image: Image.Image) -> ov.Tensor:
    """Convert a PIL image to an NHWC OpenVINO tensor."""
    return ov.Tensor(np.asarray(image.convert("RGB"), dtype=np.uint8)[None])


def stitch_images(
    images: Sequence[Image.Image],
    layout: str = "auto",
    fit: str = "pad",
    background: tuple[int, int, int] = (255, 255, 255),
    target_pixels: int = TARGET_CONDITION_PIXELS,
    max_images: int = MAX_INPUT_IMAGES,
) -> Image.Image:
    """Stitch several images into one condition image.

    Pure helper (no OpenVINO/Tkinter dependency) so it can be unit-tested on its own.

    :param images: Input images; one to ``max_images`` are accepted.
    :param layout: ``"auto"`` (near-square uniform grid), ``"horizontal"`` or ``"vertical"``.
    :param fit: ``"pad"`` (letterbox) or ``"crop"`` (center-crop fill); only used by the
        ``"auto"`` grid.
    :param background: RGB fill color for padding and empty grid cells.
    :param target_pixels: Approximate total area of the composed canvas.
    :param max_images: Maximum number of input images accepted.
    :return: Composed RGB image.
    """
    count = len(images)
    if count < 1:
        raise ValueError("At least one input image is required.")
    if count > max_images:
        raise ValueError(f"At most {max_images} input images are supported, got {count}.")

    rgb = [image.convert("RGB") for image in images]
    aspects = [max(1e-6, img.width / img.height) for img in rgb]

    if layout == "horizontal":
        total = sum(aspects)
        height = max(1, round(math.sqrt(target_pixels / total))) if total > 0 else 1
        widths = [max(1, round(aspect * height)) for aspect in aspects]
        canvas = Image.new("RGB", (sum(widths), height), background)
        x = 0
        for img, width in zip(rgb, widths):
            canvas.paste(img.resize((width, height), Image.LANCZOS), (x, 0))
            x += width
        return canvas

    if layout == "vertical":
        total = sum(1.0 / aspect for aspect in aspects)
        width = max(1, round(math.sqrt(target_pixels / total))) if total > 0 else 1
        heights = [max(1, round(width / aspect)) for aspect in aspects]
        canvas = Image.new("RGB", (width, sum(heights)), background)
        y = 0
        for img, height in zip(rgb, heights):
            canvas.paste(img.resize((width, height), Image.LANCZOS), (0, y))
            y += height
        return canvas

    # "auto": choose a near-square uniform grid that best matches the median aspect ratio.
    sorted_aspects = sorted(aspects)
    mid = count // 2
    if count % 2:
        median_aspect = sorted_aspects[mid]
    else:
        median_aspect = (sorted_aspects[mid - 1] + sorted_aspects[mid]) / 2.0
    cell_aspect = min(4.0, max(0.25, median_aspect))

    # Iterate from the widest grid down so equal scores prefer a wider (landscape)
    # layout, matching the horizontal concatenation convention of Qwen-Image-Edit.
    best_cols = 1
    best_score: float | None = None
    for cols in range(count, 0, -1):
        rows = math.ceil(count / cols)
        score = abs(math.log((cols / rows) * cell_aspect)) + 0.15 * (rows * cols - count)
        if best_score is None or score < best_score:
            best_score = score
            best_cols = cols
    cols = best_cols
    rows = math.ceil(count / cols)

    cell_h = max(1, round(math.sqrt(target_pixels / (rows * cols * cell_aspect))))
    cell_w = max(1, round(cell_aspect * cell_h))
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), background)

    if fit not in ("pad", "crop"):
        fit = "pad"
    for index, img in enumerate(rgb):
        row = index // cols
        col = index % cols
        x = col * cell_w
        y = row * cell_h
        if fit == "crop":
            tile = ImageOps.fit(img, (cell_w, cell_h), method=Image.LANCZOS)
            canvas.paste(tile, (x, y))
        else:
            tile = img.copy()
            tile.thumbnail((cell_w, cell_h), Image.LANCZOS)
            paste_x = x + (cell_w - tile.width) // 2
            paste_y = y + (cell_h - tile.height) // 2
            canvas.paste(tile, (paste_x, paste_y))
    return canvas


class QwenImageApp:
    def __init__(
        self,
        root: tk.Tk,
        model_dirs: dict[str, str],
        default_model: str,
        devices: list[str],
        default_device: str,
        available: list[str],
    ) -> None:
        self.root = root
        self.model_dirs = model_dirs
        self.available = available
        self.pipe: Any = None
        self.pipe_key: tuple[str, str] | None = None
        self.busy = False
        self.photo: Any = None
        self.preview_photo: Any = None
        self.result_photo: Any = None
        self.result_image: Image.Image | None = None
        self.input_images: list[tuple[str, Image.Image]] = []
        self.rows: list[ttk.Frame] = []
        self.thumb_refs: list[Any] = []
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()

        root.title("Qwen-Image 2.1 - Image to Image (OpenVINO)")
        root.geometry("1160x1000")

        ttk.Label(
            root,
            text="Input images (up to 10). Refer to them in the prompt as image 1, image 2, ...",
        ).pack(anchor="w", padx=8, pady=(8, 2))

        actions = ttk.Frame(root)
        actions.pack(fill="x", padx=8, pady=4)
        self.add_button = ttk.Button(actions, text="Add images...", command=self.on_add_images)
        self.add_button.pack(side="left")
        self.clear_button = ttk.Button(actions, text="Clear all", command=self.on_clear)
        self.clear_button.pack(side="left", padx=6)
        self.counter = tk.StringVar(value=f"0 / {MAX_INPUT_IMAGES}")
        ttk.Label(actions, textvariable=self.counter).pack(side="left", padx=8)

        list_frame = ttk.LabelFrame(root, text="Loaded images")
        list_frame.pack(fill="x", padx=8, pady=4)
        self.canvas = tk.Canvas(list_frame, height=160, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.rows_frame = ttk.Frame(self.canvas)
        self._rows_window = self.canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.rows_frame.bind(
            "<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self._rows_window, width=event.width),
        )

        preview_frame = ttk.LabelFrame(root, text="Condition preview")
        preview_frame.pack(fill="x", padx=8, pady=4)
        self.preview_label = ttk.Label(
            preview_frame, anchor="center", text="Add images to build a condition"
        )
        self.preview_label.pack(fill="both", expand=True, padx=4, pady=4)

        stitch_row = ttk.Frame(root)
        stitch_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(stitch_row, text="Layout").grid(row=0, column=0, sticky="e", padx=4)
        self.layout_var = tk.StringVar(value="Auto grid")
        self.layout_combo = ttk.Combobox(
            stitch_row,
            textvariable=self.layout_var,
            values=["Auto grid", "Horizontal", "Vertical"],
            width=12,
            state="readonly",
        )
        self.layout_combo.grid(row=0, column=1)
        ttk.Label(stitch_row, text="Fit").grid(row=0, column=2, sticky="e", padx=8)
        # "Fit" only affects the auto grid; horizontal/vertical always preserve aspect.
        self.fit_var = tk.StringVar(value="Pad")
        self.fit_combo = ttk.Combobox(
            stitch_row,
            textvariable=self.fit_var,
            values=["Pad", "Crop"],
            width=8,
            state="readonly",
        )
        self.fit_combo.grid(row=0, column=3)
        self.layout_combo.bind("<<ComboboxSelected>>", self._on_stitch_change)
        self.fit_combo.bind("<<ComboboxSelected>>", self._on_stitch_change)

        ttk.Label(root, text="Prompt").pack(anchor="w", padx=8, pady=(6, 0))
        self.prompt = tk.Text(root, height=5, wrap="word")
        self.prompt.pack(fill="x", padx=8, pady=4)
        self.prompt.insert("1.0", DEFAULT_PROMPT)

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
        self.actual_device = tk.StringVar(value="actual: -")
        ttk.Label(model_row, textvariable=self.actual_device).grid(row=0, column=4, padx=8)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        self.device_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        gen_row = ttk.Frame(root)
        gen_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(gen_row, text="Steps").grid(row=0, column=0, sticky="e", padx=4)
        self.steps = tk.IntVar(value=DEFAULT_STEPS)
        ttk.Spinbox(gen_row, from_=1, to=80, width=5, textvariable=self.steps).grid(row=0, column=1)
        ttk.Label(gen_row, text="Seed").grid(row=0, column=2, sticky="e", padx=8)
        self.seed = tk.StringVar(value="42")
        ttk.Entry(gen_row, textvariable=self.seed, width=12).grid(row=0, column=3)
        self.randomize_seed = tk.BooleanVar(value=False)
        ttk.Checkbutton(gen_row, text="Randomize seed", variable=self.randomize_seed).grid(
            row=0, column=4, padx=8
        )

        action = ttk.Frame(root)
        action.pack(fill="x", padx=8, pady=4)
        self.generate_button = ttk.Button(action, text="Generate", command=self.on_generate)
        self.generate_button.pack(side="left")
        self.status = tk.StringVar(value="Ready. The model loads on the first generation.")
        ttk.Label(action, textvariable=self.status).pack(side="left", padx=12)

        self.image_label = ttk.Label(root, anchor="center", text="Generated image appears here")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=4)

        save_row = ttk.Frame(root)
        save_row.pack(fill="x", padx=8, pady=4)
        self.save_button = ttk.Button(
            save_row, text="Save result...", command=self.on_save, state="disabled"
        )
        self.save_button.pack(side="left")

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ input list
    def load_paths(self, paths: Sequence[str]) -> None:
        """Load images from disk, stopping at the image limit and ignoring unreadable files."""
        overflow = 0
        for path in paths:
            if len(self.input_images) >= MAX_INPUT_IMAGES:
                overflow += 1
                continue
            try:
                image = Image.open(path).convert("RGB")
            except Exception as exc:  # unreadable/corrupt file: report and continue
                self.status.set(f"Could not open {os.path.basename(path)}: {exc}")
                continue
            self.input_images.append((path, image))
        self._rebuild_rows()
        if overflow:
            self.status.set(
                f"Reached the {MAX_INPUT_IMAGES}-image limit; ignored {overflow} extra image(s)."
            )

    def on_add_images(self) -> None:
        if self.busy:
            return
        paths = filedialog.askopenfilenames(
            title="Select input images",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self.load_paths(list(paths))

    def on_remove(self, index: int) -> None:
        if self.busy:
            return
        if 0 <= index < len(self.input_images):
            self.input_images.pop(index)
            self._rebuild_rows()

    def on_clear(self) -> None:
        if self.busy:
            return
        self.input_images.clear()
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        for row in self.rows:
            row.destroy()
        self.rows = []
        self.thumb_refs = []
        for index, (path, image) in enumerate(self.input_images):
            row = ttk.Frame(self.rows_frame)
            row.pack(fill="x", pady=2)
            thumb = image.copy()
            thumb.thumbnail((56, 56), Image.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            self.thumb_refs.append(photo)  # keep a reference so Tk does not garbage-collect it
            thumb_label = ttk.Label(row, image=photo)
            thumb_label.pack(side="left")
            ttk.Label(row, text=f"{index + 1}. {os.path.basename(path)}").pack(side="left", padx=8)
            ttk.Button(row, text="Remove", command=lambda i=index: self.on_remove(i)).pack(
                side="right"
            )
            self.rows.append(row)
        self.counter.set(f"{len(self.input_images)} / {MAX_INPUT_IMAGES}")
        self._refresh_preview()

    # ------------------------------------------------------------------ preview
    def _build_condition(self) -> Image.Image:
        layout_map = {"Auto grid": "auto", "Horizontal": "horizontal", "Vertical": "vertical"}
        fit_map = {"Pad": "pad", "Crop": "crop"}
        return stitch_images(
            [image for _, image in self.input_images],
            layout=layout_map.get(self.layout_var.get(), "auto"),
            fit=fit_map.get(self.fit_var.get(), "pad"),
        )

    def _refresh_preview(self) -> None:
        if not self.input_images:
            self.preview_photo = None
            self.preview_label.configure(image="", text="Add images to build a condition")
            return
        try:
            condition = self._build_condition()
        except Exception as exc:
            self.preview_photo = None
            self.preview_label.configure(image="", text=f"Preview error: {exc}")
            return
        preview = condition.copy()
        preview.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo, text="")

    def _on_stitch_change(self, _event: Any = None) -> None:
        self._refresh_preview()

    def _on_model_change(self, _event: Any = None) -> None:
        if not self.busy:
            self.actual_device.set("actual: -")
            self.status.set(f"Model/device changed. Next Generate reloads {self.model.get()}.")

    # ------------------------------------------------------------------ generate
    def on_generate(self) -> None:
        if self.busy:
            return
        if not self.input_images:
            self.status.set("Add at least one input image first.")
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
        layout_map = {"Auto grid": "auto", "Horizontal": "horizontal", "Vertical": "vertical"}
        fit_map = {"Pad": "pad", "Crop": "crop"}
        config = {
            "model_key": model_key,
            "model_dir": model_dir,
            "device": self.device.get(),
            "prompt": prompt,
            "steps": self.steps.get(),
            "seed": seed,
            "images": [image for _, image in self.input_images],
            "layout": layout_map.get(self.layout_var.get(), "auto"),
            "fit": fit_map.get(self.fit_var.get(), "pad"),
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
        self.add_button.state(["!disabled"] if enabled else ["disabled"])
        self.clear_button.state(["!disabled"] if enabled else ["disabled"])
        state = "readonly" if enabled else "disabled"
        self.model_combo.configure(state=state)
        self.device_combo.configure(state=state)

    def _worker(self, config: dict[str, Any]) -> None:
        try:
            effective = self._ensure_pipeline(config["model_dir"], config["device"])
            self.events.put(("device", effective))
            self.events.put(("status", f"Building condition image on {effective}..."))
            condition = stitch_images(config["images"], layout=config["layout"], fit=config["fit"])
            self.events.put(("status", f"Generating on {effective}..."))
            image = self._generate(config, condition)
            payload = (image, int(config["seed"]), config, effective, condition.size)
            self.events.put(("image", payload))
        except Exception as exc:  # surface model/runtime errors in the status bar
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _ensure_pipeline(self, model_dir: str, requested_device: str) -> str:
        """Load or reuse a pipeline and return the device it actually runs on."""
        effective = resolve_device(requested_device, self.available)
        key = (model_dir, effective)
        if self.pipe is not None and self.pipe_key == key:
            return effective

        self.pipe = None
        self.pipe_key = None
        gc.collect()
        self.events.put(("status", f"Loading {os.path.basename(model_dir)} on {effective}..."))
        try:
            pipe = ov_genai.Image2ImagePipeline(model_dir, effective)
        except Exception:
            if effective == "CPU":
                raise
            effective = "CPU"
            self.events.put(("status", "Requested device failed; falling back to CPU..."))
            pipe = ov_genai.Image2ImagePipeline(model_dir, "CPU")
        self.pipe = pipe
        self.pipe_key = (model_dir, effective)
        return effective

    def _generate(self, config: dict[str, Any], condition: Image.Image) -> Image.Image:
        output = self.pipe.generate(
            config["prompt"],
            image_to_tensor(condition),
            guidance_scale=1.0,
            num_inference_steps=int(config["steps"]),
            generator=ov_genai.TorchGenerator(int(config["seed"])),
        )
        return output_to_image(output)

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status.set(str(payload))
                elif kind == "device":
                    self.actual_device.set(f"actual: {payload}")
                elif kind == "image":
                    image, seed, config, effective, condition_size = payload
                    self._show(image)
                    self.seed.set(str(seed))  # keep the seed that was actually used
                    self.actual_device.set(f"actual: {effective}")
                    self.status.set(
                        f"Done. model={config['model_key']} "
                        f"requested={config['device']} actual={effective} "
                        f"seed={seed} steps={config['steps']} "
                        f"images={len(config['images'])} "
                        f"condition={condition_size[0]}x{condition_size[1]}"
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
        self.result_image = image
        preview = image.copy()
        preview.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
        self.result_photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.result_photo, text="")
        self.save_button.state(["!disabled"])

    def on_save(self) -> None:
        if self.result_image is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save result",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile="qwen-image-2.1-i2i.png",
        )
        if not path:
            return
        self.result_image.save(path)
        self.status.set(f"Saved result to {path}")

    def _on_close(self) -> None:
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image 2.1 image-to-image Tkinter demo.")
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
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Image paths to preload (capped at the built-in image limit).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text to prefill the editing instruction box.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dirs = dict(MODEL_DIRS)
    default_model = DEFAULT_MODEL
    if args.model:
        model_dirs["Custom"] = args.model
        default_model = "Custom"

    available = available_devices()
    devices = list(dict.fromkeys(["AUTO", *available]))
    default_device = args.device or "AUTO"
    if default_device not in devices:
        devices.append(default_device)

    root = tk.Tk()
    app = QwenImageApp(root, model_dirs, default_model, devices, default_device, available)
    if args.prompt:
        app.prompt.delete("1.0", "end")
        app.prompt.insert("1.0", args.prompt)
    if args.images:
        app.load_paths(list(args.images))
    root.mainloop()


if __name__ == "__main__":
    main()
