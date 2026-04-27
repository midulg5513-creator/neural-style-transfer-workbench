from __future__ import annotations

from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from neural_style.mask import blend_with_mask, load_mask_tensor, normalize_mask_tensor


def test_normalize_mask_tensor_reduces_rgb_mask_to_single_channel() -> None:
    mask = torch.tensor(
        [
            [
                [[0.0, 1.5], [0.5, -1.0]],
                [[0.0, 0.5], [0.5, 0.0]],
                [[1.0, 0.5], [0.5, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    normalized = normalize_mask_tensor(mask)

    assert normalized.shape == (1, 1, 2, 2)
    assert torch.all(normalized >= 0.0)
    assert torch.all(normalized <= 1.0)


def test_blend_with_mask_combines_stylized_and_content_regions() -> None:
    stylized = torch.ones(1, 3, 2, 2)
    content = torch.zeros(1, 3, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [0.25, 0.75]]]])

    blended = blend_with_mask(stylized, content, mask)

    expected = mask.expand_as(stylized)
    assert torch.allclose(blended, expected)


def test_blend_with_mask_rejects_spatial_mismatch() -> None:
    stylized = torch.ones(1, 3, 4, 4)
    content = torch.zeros(1, 3, 4, 4)
    mask = torch.ones(1, 1, 2, 2)

    with pytest.raises(ValueError, match="HxW"):
        blend_with_mask(stylized, content, mask)


def test_load_mask_tensor_forces_mask_to_requested_shape(tmp_path) -> None:
    source_path = tmp_path / "mask.png"
    Image.new("L", (90, 30), color=255).save(source_path)

    mask = load_mask_tensor(source_path, target_shape=(24, 40), device="cpu")

    assert mask.shape == (1, 1, 24, 40)
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)
