from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

import app_gui
from neural_style.config import (
    DEFAULT_CONTENT_BLEND,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_STEPS,
    DEFAULT_STYLE_STRENGTH,
    ENHANCED_MODE_CONTENT_BLEND,
    ENHANCED_MODE_NUM_STEPS,
    ENHANCED_MODE_STYLE_STRENGTH,
    MIN_NUM_STEPS,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_builds_with_simplified_header_and_controls(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(app_gui, "is_cuda_ready", lambda: True)
    monkeypatch.setattr(
        app_gui,
        "build_startup_status_message",
        lambda: "已检测到 CUDA 环境，可以按仅 GPU 模式运行。",
    )

    window = app_gui.MainWindow()
    try:
        qapp.processEvents()
        assert window.windowTitle() == "风格迁移工作台"
        assert window.top_bar_title_label.text() == "风格迁移工作台"
        assert window.workspace_title_label.text() == "预览与结果"
        assert window.run_button.isEnabled() is True
        assert window.enhanced_mode_checkbox.text() == ""
        assert window.keep_color_checkbox.text() == ""
        assert window.histogram_loss_checkbox.text() == ""
        assert window.enhanced_preset_button.text() == "强化预设"
        assert window.backbone_combo.currentData() == "vgg19"
        assert window.backbone_combo.currentText().startswith("VGG19")
        assert window.reset_parameters_button.text() == "恢复默认"
        assert hasattr(window, "paper_mode_checkbox") is False
        assert hasattr(window, "paper_preset_button") is False
        assert hasattr(window, "top_bar_subtitle_label") is False
        assert hasattr(window, "top_bar_meta_label") is False
        assert hasattr(window, "workspace_subtitle_label") is False
        assert window.mode_value_label.text() == "标准模式"
        assert "输出：" in window.plan_details_label.text()
        assert "CUDA 环境就绪" in window.environment_label.text()
    finally:
        window.close()


def test_enhanced_preset_updates_form_values(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(app_gui, "is_cuda_ready", lambda: True)
    monkeypatch.setattr(
        app_gui,
        "build_startup_status_message",
        lambda: "已检测到 CUDA 环境，可以按仅 GPU 模式运行。",
    )

    window = app_gui.MainWindow()
    try:
        window.steps_spin.setValue(MIN_NUM_STEPS)
        window.style_strength_spin.setValue(1.0)
        window.content_blend_spin.setValue(0.3)
        window.keep_color_checkbox.setChecked(True)

        window.enhanced_preset_button.click()
        qapp.processEvents()

        assert window.enhanced_mode_checkbox.isChecked() is True
        assert window.steps_spin.value() == ENHANCED_MODE_NUM_STEPS
        assert window.style_strength_spin.value() == ENHANCED_MODE_STYLE_STRENGTH
        assert window.content_blend_spin.value() == ENHANCED_MODE_CONTENT_BLEND
        assert window.keep_color_checkbox.isChecked() is False
        assert window.histogram_loss_checkbox.isChecked() is False
        assert window.mode_value_label.text() == "强化模式"
    finally:
        window.close()


def test_live_summary_and_reset_button_reflect_current_plan(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(app_gui, "is_cuda_ready", lambda: True)
    monkeypatch.setattr(
        app_gui,
        "build_startup_status_message",
        lambda: "已检测到 CUDA 环境，可以按仅 GPU 模式运行。",
    )

    window = app_gui.MainWindow()
    try:
        window.content_input.setText(r"examples\content.png")
        window.style_input.setText(r"examples\style.png")
        window.mask_input.setText(r"examples\mask.png")
        window.output_input.setText(r"outputs\custom-run")
        window.steps_spin.setValue(640)
        window.style_strength_spin.setValue(1.8)
        window.content_blend_spin.setValue(0.1)
        window.image_size_spin.setValue(1024)
        window.backbone_combo.setCurrentIndex(window.backbone_combo.findData("resnet50"))
        window.keep_color_checkbox.setChecked(True)
        window.histogram_loss_checkbox.setChecked(True)
        window.enhanced_mode_checkbox.setChecked(True)
        qapp.processEvents()

        assert window.mode_value_label.text() == "强化模式"
        assert window.size_value_label.text() == "1024 px"
        assert window.control_value_label.text() == "骨干 ResNet50 | 保色 开 | 遮罩 开 | 直方图 开"
        assert "骨干：resnet50" in window.plan_details_label.text()
        assert "custom-run.png" in window.plan_details_label.text()
        assert "custom-run.json" in window.plan_details_label.text()
        assert "强化模式" in window.action_hint_label.text()
        assert "ResNet50" in window.action_hint_label.text()

        window.reset_parameters_button.click()
        qapp.processEvents()

        assert window.steps_spin.value() == DEFAULT_NUM_STEPS
        assert window.style_strength_spin.value() == DEFAULT_STYLE_STRENGTH
        assert window.content_blend_spin.value() == DEFAULT_CONTENT_BLEND
        assert window.image_size_spin.value() == DEFAULT_IMAGE_SIZE
        assert window.keep_color_checkbox.isChecked() is False
        assert window.histogram_loss_checkbox.isChecked() is False
        assert window.enhanced_mode_checkbox.isChecked() is False
        assert window.backbone_combo.currentData() == "vgg19"
        assert window.mode_value_label.text() == "标准模式"
    finally:
        window.close()
