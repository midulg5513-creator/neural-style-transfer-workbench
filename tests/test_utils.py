from __future__ import annotations

import sys
import types

from PIL import Image
import pytest

from neural_style.utils import (
    build_output_paths,
    calculate_resize_shape,
    load_image_tensor,
    load_rgb_image,
    save_tensor_image,
    tensor_to_pil_image,
)


class FakeDevice:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class FakeTensor:
    def __init__(self, array, device: str = "cpu") -> None:
        import numpy as np

        self.array = np.array(array, copy=True)
        self.device = device

    @property
    def shape(self):
        return self.array.shape

    def dim(self) -> int:
        return self.array.ndim

    def to(self, device) -> "FakeTensor":
        return FakeTensor(self.array, device=str(device))

    def permute(self, *dims: int) -> "FakeTensor":
        import numpy as np

        return FakeTensor(np.transpose(self.array, dims), device=self.device)

    def contiguous(self) -> "FakeTensor":
        return FakeTensor(self.array, device=self.device)

    def unsqueeze(self, dim: int) -> "FakeTensor":
        import numpy as np

        return FakeTensor(np.expand_dims(self.array, axis=dim), device=self.device)

    def detach(self) -> "FakeTensor":
        return FakeTensor(self.array, device=self.device)

    def cpu(self) -> "FakeTensor":
        return FakeTensor(self.array, device="cpu")

    def clamp(self, low: float, high: float) -> "FakeTensor":
        import numpy as np

        return FakeTensor(np.clip(self.array, low, high), device=self.device)

    def numpy(self):
        import numpy as np

        return np.array(self.array, copy=True)

    def __getitem__(self, index) -> "FakeTensor":
        return FakeTensor(self.array[index], device=self.device)


def install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    fake_torch = types.SimpleNamespace(
        Tensor=FakeTensor,
        device=lambda value: FakeDevice(str(value)),
        from_numpy=lambda array: FakeTensor(np.array(array, copy=True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_calculate_resize_shape_preserves_aspect_ratio() -> None:
    assert calculate_resize_shape(120, 60, target_size=48) == (48, 24)
    assert calculate_resize_shape(60, 120, target_size=48) == (24, 48)


def test_build_output_paths_defaults_to_outputs_dir() -> None:
    image_path, metadata_path = build_output_paths()

    assert image_path.name == "result.png"
    assert image_path.parent.name == "outputs"
    assert metadata_path.name == "result.json"


def test_build_output_paths_adds_default_suffix(tmp_path) -> None:
    image_path, metadata_path = build_output_paths(tmp_path / "styled-output")

    assert image_path.suffix == ".png"
    assert metadata_path == image_path.with_suffix(".json")


def test_load_rgb_image_converts_to_rgb_and_resizes(tmp_path) -> None:
    source_path = tmp_path / "input.png"
    Image.new("RGBA", (80, 40), color=(10, 20, 30, 128)).save(source_path)

    image = load_rgb_image(source_path, target_size=32)

    assert image.mode == "RGB"
    assert image.size == (32, 16)


def test_load_image_tensor_moves_to_requested_device(tmp_path, monkeypatch) -> None:
    install_fake_torch(monkeypatch)

    source_path = tmp_path / "input.png"
    Image.new("RGB", (100, 50), color=(64, 128, 255)).save(source_path)

    tensor = load_image_tensor(source_path, target_size=40, device="cpu")

    assert tensor.shape == (1, 3, 20, 40)
    assert tensor.device == "cpu"


def test_tensor_to_pil_image_clamps_batch_tensor(monkeypatch) -> None:
    install_fake_torch(monkeypatch)

    tensor = FakeTensor(
        [
            [
                [[-0.5, 0.5], [1.5, 0.25]],
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.9, 1.2], [0.0, -1.0]],
            ]
        ]
    )

    image = tensor_to_pil_image(tensor)

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (0, 26, 230)
    assert image.getpixel((1, 0)) == (128, 51, 255)


def test_save_tensor_image_writes_file(tmp_path, monkeypatch) -> None:
    install_fake_torch(monkeypatch)

    tensor = FakeTensor(
        [
            [
                [[0.0, 1.0], [0.5, 0.25]],
                [[0.0, 0.0], [0.5, 0.25]],
                [[1.0, 0.0], [0.5, 0.25]],
            ]
        ]
    )

    output_path = save_tensor_image(tensor, tmp_path / "saved-result")

    assert output_path.exists()
    assert output_path.suffix == ".png"
