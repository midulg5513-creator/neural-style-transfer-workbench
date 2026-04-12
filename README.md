# Neural Style Transfer App

Computer vision final-project desktop application for neural style transfer.

## Current Scope

- Local desktop application built with PySide6
- Neural style transfer based on pretrained VGG19 features
- Required feature extensions:
  - `style_strength`
  - `keep_color`
  - `mask`
- Output image plus JSON metadata sidecar for each run

## Runtime Requirement

This project is intentionally scoped for a CUDA-capable NVIDIA machine.
CPU fallback is not part of the current plan.

Before running the app, confirm that:

1. PyTorch is installed with a CUDA-enabled build.
2. `torch.cuda.is_available()` returns `True`.
3. The target machine is the same prepared environment intended for the final demo.

## Offline Demo Preparation

The final demo must not depend on live downloads during presentation.
Prepare the environment ahead of time by:

1. Installing project dependencies in a local `.venv`.
2. Pre-downloading the pretrained VGG19 weights on the target machine.
3. Verifying that the cached weights remain available offline.
4. Running a smoke test before demo day.

Detailed execution commands and warm-up instructions will be added as the CLI and core pipeline are implemented.

## Repository Layout

```text
.
├─ app_gui.py
├─ cli.py
├─ neural_style/
├─ examples/
├─ outputs/
└─ tests/
```

## Outputs

Generated result images should be written to `outputs/`.
Each output image should have a matching JSON metadata file that records the input files, parameters, device context, and output location.
