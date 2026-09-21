# Qwen-Image 2.1 image generation with OpenVINO

Qwen-Image 2.1 is a unified image generation model that supports text-to-image generation and image-conditioned editing in one pipeline. The prompt and condition images are encoded together by Qwen3-VL and processed by a block-causal diffusion transformer.

For model architecture and usage details, see the [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) model card and the [Qwen-Image source repository](https://github.com/QwenLM/Qwen-Image).

This tutorial demonstrates how to:

- load a pre-exported OpenVINO IR of [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) from a local directory;
- run text-to-image generation with OpenVINO GenAI;
- run image-conditioned editing with the same exported model;
- save generated images with reproducible configuration details in their filenames;
- measure pipeline loading, first-run, and warm-run latency;
- launch an interactive demo with pipeline, precision, and device selection.

> **Important:** Qwen-Image 2.1 image conditioning is not classic image-to-image generation based on adding noise to an initial image. The condition image is part of the multimodal context, so this notebook intentionally does not expose a `strength` parameter.

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

## Notebook Contents

1. Install the latest stable Gradio, PyTorch, and utility packages together with OpenVINO, OpenVINO Tokenizers, and OpenVINO GenAI nightly builds
2. Select the model root and the FP16 or INT4 precision
3. Run text-to-image generation
4. Run image-conditioned editing
5. Benchmark both scenarios
6. Launch an interactive demo with dynamic pipeline, precision, and device selection

## Model Directory

The notebook expects OpenVINO IR directories exported beforehand by an Optimum Intel / OpenVINO GenAI build that supports Qwen-Image 2.1. Each directory must contain `model_index.json` (with `"_class_name": "QwenImage21Pipeline"`) and the `processor`, `scheduler`, `text_encoder`, `text_encoder_i2i`, `transformer`, `vae_encoder`, `vae_decoder`, and `vision_encoder` subdirectories.

One directory is expected per weight precision, named `Qwen-Image-2.1-IR-<precision>`:

| Precision | Directory |
| --- | --- |
| FP16 | `<Model root>/Qwen-Image-2.1-IR-FP16` |
| INT4 | `<Model root>/Qwen-Image-2.1-IR-INT4` |

The default model root is `C:\openvino` and can be overridden with the `QWEN_IMAGE_21_OV_DIR` environment variable or the notebook widget.

## Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to the [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/qwen-image-2.1/README.md" />
