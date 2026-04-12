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
            "keep_color": True,
        },
        device="cpu",
    )

    payload = metadata.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["inputs"]["content_image"].endswith("content.png")
    assert payload["inputs"]["mask_image"].endswith("mask.png")
    assert payload["parameters"]["num_steps"] == 300
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
            "mask_path": None,
        },
        device="cpu",
    )

    saved_path = save_run_metadata(metadata)

    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_path.exists()
    assert payload["inputs"]["mask_image"] is None
    assert payload["parameters"]["style_strength"] == 0.8
    assert payload["runtime"]["torch_available"] is False
