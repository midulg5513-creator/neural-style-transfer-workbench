from __future__ import annotations

import json

from neural_style.metadata import build_run_metadata, save_run_metadata


def test_build_run_metadata_contains_expected_sections(tmp_path) -> None:
    metadata = build_run_metadata(
        content_path=tmp_path / "content.png",
        style_path=tmp_path / "style.png",
        mask_path=tmp_path / "mask.png",
        output_image_path=tmp_path / "result.png",
        metadata_path=tmp_path / "result.json",
        parameters={
            "num_steps": 300,
            "style_strength": 1.2,
            "content_blend": 0.15,
            "tv_weight": 1e-4,
            "histogram_weight": 1.0,
            "histogram_loss": True,
            "init_mode": "content_noise",
            "use_avg_pool": True,
            "enhanced_mode": True,
            "paper_mode": False,
            "keep_color": True,
            "backbone": "vgg19",
            "layer_preset": "legacy",
            "scale_schedule": [768],
        },
        device="cpu",
    )

    payload = metadata.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["inputs"]["content_image"].endswith("content.png")
    assert payload["inputs"]["mask_image"].endswith("mask.png")
    assert payload["parameters"]["num_steps"] == 300
    assert payload["parameters"]["content_blend"] == 0.15
    assert payload["parameters"]["init_mode"] == "content_noise"
    assert payload["parameters"]["histogram_weight"] == 1.0
    assert payload["parameters"]["histogram_loss"] is True
    assert payload["parameters"]["use_avg_pool"] is True
    assert payload["parameters"]["enhanced_mode"] is True
    assert payload["parameters"]["paper_mode"] is False
    assert payload["parameters"]["backbone"] == "vgg19"
    assert payload["parameters"]["layer_preset"] == "legacy"
    assert payload["parameters"]["scale_schedule"] == [768]
    assert payload["runtime"]["device"] == "cpu"
    assert payload["artifacts"]["metadata_file"].endswith("result.json")


def test_save_run_metadata_writes_json_sidecar(tmp_path) -> None:
    metadata = build_run_metadata(
        content_path=tmp_path / "content.png",
        style_path=tmp_path / "style.png",
        output_image_path=tmp_path / "result.png",
        metadata_path=tmp_path / "result.json",
        parameters={
            "num_steps": 250,
            "style_strength": 0.8,
            "content_blend": 0.2,
            "tv_weight": 0.0,
            "histogram_weight": 0.0,
            "histogram_loss": False,
            "init_mode": "content",
            "use_avg_pool": False,
            "enhanced_mode": False,
            "paper_mode": True,
            "mask_path": None,
            "backbone": "resnet50",
            "layer_preset": "paper",
            "scale_schedule": [256, 512],
        },
        device="cpu",
    )

    saved_path = save_run_metadata(metadata)

    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_path.exists()
    assert payload["inputs"]["mask_image"] is None
    assert payload["parameters"]["style_strength"] == 0.8
    assert payload["parameters"]["content_blend"] == 0.2
    assert payload["parameters"]["histogram_weight"] == 0.0
    assert payload["parameters"]["histogram_loss"] is False
    assert payload["parameters"]["enhanced_mode"] is False
    assert payload["parameters"]["paper_mode"] is True
    assert payload["parameters"]["backbone"] == "resnet50"
    assert payload["parameters"]["layer_preset"] == "paper"
    assert payload["parameters"]["scale_schedule"] == [256, 512]
    assert payload["runtime"]["device"] == "cpu"
    assert isinstance(payload["runtime"]["torch_available"], bool)
