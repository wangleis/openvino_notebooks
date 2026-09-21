"""Pure text-to-image demo for Qwen-Image 2.1 with a Tkinter window.

Type a prompt and press Generate; the image is shown in the window (not saved).

Run inside a venv that already has the Qwen-Image 2.1 stack:
    torch>=2.4, transformers>=5.17, diffusers @ git main, accelerate, pillow
    python qwen_image_t2i_tk.py
"""

from __future__ import annotations

import queue
import random
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

import torch
from PIL import Image, ImageTk

# Repo id or a local path to the downloaded Qwen-Image 2.1 model.
MODEL_ID = "Qwen/Qwen-Image-2.1"
RESOLUTIONS = {
    "1024 x 1024": (1024, 1024),
    "1536 x 1536": (1536, 1536),
    "2048 x 2048 (native)": (2048, 2048),
    "2048 x 1152 (16:9)": (2048, 1152),
    "1152 x 2048 (9:16)": (1152, 2048),
}
DEFAULT_STEPS = 40
MAX_SEED = 2**31 - 1
PREVIEW_MAX = 768


def pick_device() -> str:
    """Use CUDA when available, otherwise fall back to CPU (slow)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class QwenImageApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.device = pick_device()
        self.pipe: Any = None
        self.busy = False
        self.photo: Any = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()

        root.title(f"Qwen-Image 2.1 - Text to Image ({self.device.upper()})")
        root.geometry("820x900")

        ttk.Label(root, text="Prompt").pack(anchor="w", padx=8, pady=4)
        self.prompt = tk.Text(root, height=5, wrap="word")
        self.prompt.pack(fill="x", padx=8, pady=4)
        self.prompt.insert(
            "1.0",
            'A cozy coffee shop with a chalkboard sign reading "Qwen Coffee", cinematic lighting',
        )

        settings = ttk.Frame(root)
        settings.pack(fill="x", padx=8, pady=4)
        for col in range(8):
            settings.columnconfigure(col, weight=0)

        ttk.Label(settings, text="Steps").grid(row=0, column=0, sticky="e", padx=4)
        self.steps = tk.IntVar(value=DEFAULT_STEPS)
        ttk.Spinbox(settings, from_=1, to=80, width=5, textvariable=self.steps).grid(row=0, column=1)

        ttk.Label(settings, text="Resolution").grid(row=0, column=2, sticky="e", padx=4)
        self.resolution = tk.StringVar(value="1024 x 1024")
        ttk.Combobox(
            settings, textvariable=self.resolution, values=list(RESOLUTIONS), width=20, state="readonly"
        ).grid(row=0, column=3)

        ttk.Label(settings, text="Seed").grid(row=0, column=4, sticky="e", padx=4)
        self.seed = tk.StringVar(value="42")
        ttk.Entry(settings, textvariable=self.seed, width=12).grid(row=0, column=5)

        self.randomize_seed = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings, text="Randomize seed", variable=self.randomize_seed).grid(
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

    def on_generate(self) -> None:
        if self.busy:
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            self.status.set("Enter a prompt first.")
            return

        seed = random.randint(0, MAX_SEED) if self.randomize_seed.get() else self._current_seed()
        try:
            width, height = RESOLUTIONS[self.resolution.get()]
        except KeyError:
            width, height = 1024, 1024

        config = {
            "prompt": prompt,
            "steps": self.steps.get(),
            "width": width,
            "height": height,
            "seed": seed,
        }
        self.busy = True
        self.generate_button.state(["disabled"])
        self.status.set("Working...")
        threading.Thread(target=self._worker, args=(config,), daemon=True).start()
        self.root.after(100, self._poll)

    def _current_seed(self) -> int:
        try:
            return int(self.seed.get())
        except ValueError:
            return 42

    def _worker(self, config: dict[str, Any]) -> None:
        try:
            self._ensure_pipeline()
            self.events.put(("status", "Generating... (40 steps can take a while)"))
            image = self._generate(config)
            self.events.put(("image", (image, int(config["seed"]))))
        except Exception as exc:  # surface any model/runtime error in the status bar
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _ensure_pipeline(self) -> None:
        if self.pipe is not None:
            return
        self.events.put(("status", f"Loading {MODEL_ID} on {self.device.upper()}... (first run downloads)"))
        from diffusers import QwenImage21Pipeline

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()  # keeps the model runnable on limited VRAM
        else:
            pipe.to("cpu")
        self.pipe = pipe

    def _generate(self, config: dict[str, Any]) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(int(config["seed"]))
        result = self.pipe(
            prompt=config["prompt"],
            height=int(config["height"]),
            width=int(config["width"]),
            num_inference_steps=int(config["steps"]),
            generator=generator,
            output_type="pil",
        )
        image = result.images[0]
        return image.convert("RGB")

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status.set(str(payload))
                elif kind == "image":
                    image, seed = payload
                    self._show(image)
                    self.seed.set(str(seed))  # keep the seed that was actually used
                    self.status.set(
                        f"Done. seed={seed} steps={self.steps.get()} {self.resolution.get()}"
                    )
                    self.busy = False
                    self.generate_button.state(["!disabled"])
                elif kind == "error":
                    self.status.set(str(payload))
                    self.busy = False
                    self.generate_button.state(["!disabled"])
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


def main() -> None:
    root = tk.Tk()
    QwenImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
