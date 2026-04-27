# 神经风格迁移工作台

一个基于 `Python + PyTorch + PySide6` 的桌面版神经风格迁移项目，提供命令行和图形界面两种使用方式。项目以 `Gatys + LBFGS` 为核心流程，支持 `VGG19` 与 `ResNet50` 两种特征骨干，并补充了保色、局部遮罩、强化模式、论文模式与元数据留档等工程化能力。

## 功能特性

- 基于 `VGG19 + Gatys + LBFGS` 的经典神经风格迁移流程
- 可切换 `VGG19` / `ResNet50` 特征骨干
- 支持 `keep_color`、`mask`、`style_strength`
- 支持 `enhanced_mode` 与 `paper_mode`
- 支持 `histogram loss` 以改善局部纹理稳定性
- 提供 `CLI` 与 `PySide6 GUI`
- 结果图像会同时生成同名 `JSON` 运行记录

## 环境要求

- Windows
- NVIDIA GPU
- 安装了 CUDA 版 PyTorch
- `torch.cuda.is_available()` 返回 `True`

当前仓库默认按 `GPU-only` 方案设计，不提供 CPU 回退。

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

如果你的环境不是 `cu126`，建议先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配的 CUDA 版 `torch` / `torchvision`，再补装其余依赖。

### 2. 检查 CUDA

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

### 3. 预热模型权重

```powershell
.\.venv\Scripts\python cli.py warmup
```

这一步会缓存 `VGG19` 和 `ResNet50` 所需权重，便于后续离线运行。

### 4. 运行 CLI

```powershell
.\.venv\Scripts\python cli.py run `
  --content .\path\to\content.png `
  --style .\path\to\style.png `
  --output .\outputs\demo-result.png `
  --steps 300 `
  --style-strength 1.0 `
  --image-size 768 `
  --backbone vgg19 `
  --keep-color
```

如果需要局部风格迁移，可以额外加入：

```powershell
  --mask .\path\to\mask.png
```

如需更接近论文风格的参数组合，可使用：

```powershell
.\.venv\Scripts\python cli.py run `
  --content .\path\to\content.png `
  --style .\path\to\style.png `
  --output .\outputs\paper-result.png `
  --steps 1000 `
  --image-size 768 `
  --paper-mode
```

### 5. 启动 GUI

```powershell
.\.venv\Scripts\python app_gui.py
```

GUI 支持选择内容图、风格图、遮罩图、输出路径和主要参数，并会在后台线程中完成生成，避免界面卡死。

## 仓库结构

```text
.
├─ app_gui.py
├─ cli.py
├─ neural_style/
├─ docs/
├─ examples/
├─ outputs/
└─ tests/
```

- `neural_style/`：核心算法、模型封装、校验与工具函数
- `docs/`：中文使用文档与实验说明
- `examples/`：示例素材约定说明
- `outputs/`：默认输出目录，仅保留占位文件
- `tests/`：自动化测试

## 输出内容

每次成功运行后，默认会在 `outputs/` 中生成：

- 一张结果图像，例如 `outputs\demo-result.png`
- 一份同名参数记录，例如 `outputs\demo-result.json`

`JSON` 中会记录输入路径、输出路径、步数、风格强度、骨干网络、运行设备等信息，便于复现实验。

## 测试

```powershell
.\.venv\Scripts\python -m pytest -q
```

## 已知限制

- 当前主要面向 Windows + NVIDIA CUDA 环境
- 依赖 GPU，不支持 CPU 回退
- 仓库默认不附带可直接运行的素材图片，使用前请自行准备内容图、风格图和可选遮罩图

## 文档

- `docs/中文操作手册与配置指南.md`
- `docs/桌面版操作手册.md`
- `docs/backbone-extension.md`
- `docs/参考论文与资料.md`

## 发布建议

在正式发布到 GitHub 前，建议再补充两项仓库元信息：

- 选择并添加合适的 `LICENSE`
- 在仓库首页或 `docs/assets/` 中放置 1 到 2 张界面或结果截图
