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

The CLI and desktop GUI are both available in the prepared CUDA environment.

## CLI Usage

Detailed Chinese setup and operation guide:

- `docs/中文操作手册与配置指南.md`

Create the local environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Warm up the pretrained VGG19 weights before the offline demo:

```powershell
.\.venv\Scripts\python cli.py warmup
```

Run one style-transfer pass and save both the image and JSON sidecar:

```powershell
.\.venv\Scripts\python cli.py run `
  --content examples\content.jpg `
  --style examples\style.jpg `
  --output outputs\demo-result.png `
  --steps 300 `
  --style-strength 1.0 `
  --image-size 768 `
  --keep-color
```

Optional local-style transfer mask:

```powershell
.\.venv\Scripts\python cli.py run `
  --content examples\content.jpg `
  --style examples\style.jpg `
  --mask examples\mask.png `
  --output outputs\masked-result.png
```

## GUI Usage

Launch the desktop interface from the prepared virtual environment:

```powershell
.\.venv\Scripts\python app_gui.py
```

GUI flow:

1. Select a content image and a style image.
2. Optionally select a mask image.
3. Choose the output image path or keep the default `outputs\result.png`.
4. Adjust steps, style strength, image size, and `keep_color`.
5. Start the run and wait for the progress bar, previews, and saved-output summary.

## GUI Troubleshooting

- If the `Start Transfer` button is disabled, CUDA is not available in the current environment.
- If the app reports `Please choose a content image.` or `Please choose a style image.`, the required input path is still empty.
- If the app reports an unsupported output suffix, use `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.webp`.
- If a run is cancelled, the GUI returns to an idle state and no new result image is kept for that cancelled job.
- If the app fails during execution, verify that the selected files are valid local images and that the CUDA-enabled PyTorch environment still reports `torch.cuda.is_available() == True`.

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
