from __future__ import annotations

from pathlib import Path

import pytest

from neural_style.validation import (
    CUDA_REQUIRED_MESSAGE,
    ValidationError,
    build_startup_status_message,
    normalize_output_path,
    require_cuda,
    validate_image_path,
    validate_image_size,
    validate_num_steps,
    validate_output_image_path,
    validate_style_strength,
)


def test_require_cuda_raises_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.validation.is_cuda_ready", lambda: False)

    with pytest.raises(ValidationError, match="支持 CUDA 的 NVIDIA 显卡"):
        require_cuda()


def test_build_startup_status_message_reports_missing_cuda(monkeypatch) -> None:
    monkeypatch.setattr("neural_style.validation.is_cuda_ready", lambda: False)

    message = build_startup_status_message()

    assert message == CUDA_REQUIRED_MESSAGE


def test_validate_image_path_rejects_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(ValidationError, match="不存在"):
        validate_image_path(missing_path, "content image")


def test_validate_image_path_rejects_empty_value() -> None:
    with pytest.raises(ValidationError, match="请选择内容图像"):
        validate_image_path("", "内容图像")


def test_validate_num_steps_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError, match="迭代步数"):
        validate_num_steps(1)


def test_validate_style_strength_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError, match="风格强度"):
        validate_style_strength(100.0)


def test_validate_image_size_rejects_nonpositive_values() -> None:
    with pytest.raises(ValidationError, match="输出尺寸"):
        validate_image_size(0)


def test_normalize_output_path_adds_png_suffix(tmp_path) -> None:
    output_path = normalize_output_path(tmp_path / "styled-output")

    assert output_path == Path(tmp_path / "styled-output.png")


def test_validate_output_image_path_rejects_unsupported_suffix(tmp_path) -> None:
    with pytest.raises(ValidationError, match="受支持的后缀"):
        validate_output_image_path(tmp_path / "styled-output.txt")
