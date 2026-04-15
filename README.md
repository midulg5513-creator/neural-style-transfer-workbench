# 神经风格迁移演示系统

计算机视觉期末项目，本地桌面形态，技术栈为 `Python + PyTorch + PySide6`。

当前已完成：

- 基于 `VGG19 + Gatys + LBFGS` 的神经风格迁移主流程
- `style_strength`、`keep_color`、`mask` 三个扩展功能
- PySide6 图形界面，支持后台线程运行、进度反馈、预览对比、结果保存
- 输出图像与同名 `JSON` 参数记录文件

## 运行前提

本项目是明确的 `仅 GPU` 方案，不提供 CPU 回退。

运行机器必须满足：

- Windows
- NVIDIA 显卡
- PyTorch CUDA 版本安装正确
- `torch.cuda.is_available()` 返回 `True`

## 已验证环境

以下版本已经在当前演示机器上实际验证通过：

- Python: `3.13.0`
- torch: `2.7.1+cu126`
- torchvision: `0.22.1+cu126`
- PySide6: `6.8.3`
- Pillow: `12.1.1`
- scikit-image: `0.25.2`
- pytest: `8.4.2`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- PyTorch CUDA build: `12.6`

## 安装步骤

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

安装完成后，先验证 CUDA 是否真的可用：

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

预期结果：

- `torch.__version__` 为 `2.7.1+cu126`
- `torch.cuda.is_available()` 为 `True`
- 能正确输出 `NVIDIA GeForce RTX 4060 Laptop GPU`

## 离线预热

答辩前必须先预热一次预训练权重：

```powershell
.\.venv\Scripts\python cli.py warmup
```

当前机器上的缓存目录已经验证为：

```text
C:\Users\33124\.cache\torch\hub\checkpoints
```

其中 `VGG19` 权重文件应存在：

```text
vgg19-dcbb9e9d.pth
```

如果这个文件没有缓存成功，不要直接进入离线演示。

## CLI 快速运行

```powershell
.\.venv\Scripts\python cli.py run `
  --content .\你的内容图.png `
  --style .\你的风格图.png `
  --output .\outputs\demo-result.png `
  --steps 300 `
  --style-strength 1.0 `
  --image-size 768 `
  --keep-color
```

如需局部风格迁移，可额外加入：

```powershell
  --mask .\你的遮罩图.png
```

## GUI 运行

```powershell
.\.venv\Scripts\python app_gui.py
```

GUI 当前已支持完整演示流程：

1. 选择内容图和风格图
2. 可选选择遮罩图
3. 设置输出图像路径
4. 调整步数、风格强度、图像尺寸、保留原色
5. 点击“开始生成”
6. 查看进度条、三块预览和结果摘要

## 输出说明

每次成功运行后，默认会在 `outputs/` 下得到两份文件：

- 结果图像，例如 `outputs\demo-result.png`
- 参数记录，例如 `outputs\demo-result.json`

JSON 会记录：

- 输入图路径
- 输出图路径
- 迭代步数
- 风格强度
- 图像尺寸
- 是否保留原色
- 运行设备与 PyTorch 信息

## 演示前冒烟清单

正式展示前建议按这个顺序检查：

1. `nvidia-smi` 能正常执行
2. `.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
3. `.\.venv\Scripts\python cli.py warmup` 成功
4. `.\.venv\Scripts\python app_gui.py` 能正常启动
5. 用一组小图做一次快速生成，确认图片和 JSON 都能落盘

推荐快速冒烟命令：

```powershell
.\.venv\Scripts\python cli.py run `
  --content .\你的内容图.png `
  --style .\你的风格图.png `
  --output .\outputs\smoke.png `
  --steps 50 `
  --style-strength 1.0 `
  --image-size 256
```

## 常见问题

- 如果“开始生成”按钮不可点，通常是 CUDA 不可用。
- 如果提示找不到内容图或风格图，先检查路径是否真实存在。
- 如果提示输出后缀不支持，请使用 `.png`、`.jpg`、`.jpeg`、`.bmp` 或 `.webp`。
- 如果第一次运行很慢，通常是 CUDA 初始化和模型权重首次加载导致，答辩前先做 `warmup`。

## 参考文档

- `docs/中文操作手册与配置指南.md`
