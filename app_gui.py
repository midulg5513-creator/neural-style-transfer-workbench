"""Desktop entry point for the PySide6 application."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from neural_style.config import (
    APP_MINIMUM_HEIGHT,
    APP_MINIMUM_WIDTH,
    APP_ORGANIZATION,
    APP_TITLE,
    APP_WINDOW_HEIGHT,
    APP_WINDOW_WIDTH,
    DEFAULT_CONTENT_BLEND,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_STEPS,
    DEFAULT_STYLE_STRENGTH,
    GUI_IMAGE_FILE_FILTER,
    GUI_MAX_IMAGE_SIZE,
    GUI_MIN_IMAGE_SIZE,
    GUI_PREVIEW_HEIGHT,
    GUI_PREVIEW_WIDTH,
    MAX_CONTENT_BLEND,
    MAX_NUM_STEPS,
    MAX_STYLE_STRENGTH,
    MIN_CONTENT_BLEND,
    MIN_NUM_STEPS,
    MIN_STYLE_STRENGTH,
    ENHANCED_MODE_CONTENT_BLEND,
    ENHANCED_MODE_NUM_STEPS,
    ENHANCED_MODE_STYLE_STRENGTH,
)
from neural_style.model import BACKBONE_LABELS, DEFAULT_BACKBONE
from neural_style.utils import default_output_path, load_preview_png_bytes, metadata_output_path
from neural_style.validation import (
    ValidationError,
    build_startup_status_message,
    is_cuda_ready,
    require_cuda,
    validate_content_blend,
    validate_image_path,
    validate_image_size,
    validate_num_steps,
    validate_optional_image_path,
    validate_output_image_path,
    validate_style_strength,
)
from neural_style.workers import (
    StyleTransferRunProgress,
    StyleTransferRunRequest,
    StyleTransferRunResult,
    StyleTransferWorker,
)


class PreviewPane(QFrame):
    """Reusable preview card used for content/style/result image areas."""

    def __init__(self, title: str, placeholder_text: str) -> None:
        super().__init__()
        self._placeholder_text = placeholder_text
        self._source_pixmap: QPixmap | None = None

        self.setObjectName("previewPane")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        layout = QVBoxLayout(self)
        self._layout = layout
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("previewTitle")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumSize(QSize(220, 160))
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.image_label.setObjectName("previewImage")

        self.caption_label = QLabel()
        self.caption_label.setWordWrap(True)
        self.caption_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.caption_label.setObjectName("previewCaption")

        layout.addWidget(title_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.caption_label)
        self.clear_preview()

    def clear_preview(self, caption: str | None = None) -> None:
        """Reset the preview back to its placeholder state."""
        self._source_pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(self._placeholder_text)
        self.caption_label.setText(caption or "尚未选择文件。")

    def set_preview_pixmap(self, pixmap: QPixmap, caption: str) -> None:
        """Show a loaded pixmap inside the preview card."""
        self._source_pixmap = pixmap
        self.image_label.setText("")
        self.caption_label.setText(caption)
        self._apply_scaled_pixmap()

    def set_preview_bytes(self, preview_bytes: bytes, caption: str) -> None:
        """Decode preview bytes and display them in the preview card."""
        pixmap = QPixmap()
        if not pixmap.loadFromData(preview_bytes, "PNG"):
            self.clear_preview("预览图数据无法解析。")
            return
        self.set_preview_pixmap(pixmap, caption)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep the displayed preview scaled with the widget size."""
        super().resizeEvent(event)
        if self._source_pixmap is not None:
            self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        """Scale the source pixmap into the preview viewport."""
        if self._source_pixmap is None:
            return
        target_size = self.image_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = QSize(GUI_PREVIEW_WIDTH, GUI_PREVIEW_HEIGHT)
        scaled = self._source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def configure_density(
        self,
        *,
        image_min_size: QSize,
        min_height: int,
        margins: tuple[int, int, int, int],
        spacing: int,
    ) -> None:
        """Tune the pane for either hero-preview or compact-reference usage."""
        self.image_label.setMinimumSize(image_min_size)
        self.setMinimumHeight(min_height)
        self._layout.setContentsMargins(*margins)
        self._layout.setSpacing(spacing)


class MainWindow(QMainWindow):
    """Primary application window for desktop style-transfer control."""

    cancel_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cuda_ready = is_cuda_ready()
        self._worker_thread: QThread | None = None
        self._worker: StyleTransferWorker | None = None

        self.setWindowTitle(APP_TITLE)
        self.resize(APP_WINDOW_WIDTH, APP_WINDOW_HEIGHT)
        self.setMinimumSize(APP_MINIMUM_WIDTH, APP_MINIMUM_HEIGHT)

        self._apply_window_style()
        self._build_ui()
        self._bind_events()
        self._refresh_environment_status()
        self._refresh_live_summary()
        self._sync_responsive_layouts()

    def _apply_window_style(self) -> None:
        """Apply a calm, ChatGPT-inspired desktop visual language."""
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f7f7f5;
            }
            QWidget {
                color: #1f1f1f;
                font-family: "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Aptos", "Segoe UI Variable Text";
                font-size: 13px;
            }
            QFrame#appShell {
                background: #ffffff;
                border: 1px solid #e7e7e3;
                border-radius: 14px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea#sidebarScroll {
                min-width: 344px;
            }
            QSplitter::handle {
                background: #efefec;
                width: 6px;
                margin: 6px 2px;
                border-radius: 3px;
            }
            QFrame#topBar {
                background: transparent;
                border: none;
                border-radius: 0;
            }
            QLabel#topBarTitle {
                color: #202123;
                font-size: 24px;
                font-weight: 700;
                letter-spacing: 0;
            }
            QFrame#sidebarSurface {
                background: #f7f7f5;
                border: 1px solid #ececea;
                border-radius: 10px;
            }
            QFrame#workspaceSurface {
                background: transparent;
            }
            QFrame#workspaceHeader {
                background: #ffffff;
                border: 1px solid #ececea;
                border-radius: 8px;
            }
            QLabel#workspaceTitle {
                color: #202123;
                font-size: 17px;
                font-weight: 700;
            }
            QFrame#insightCard {
                background: #ffffff;
                border: 1px solid #ececea;
                border-radius: 8px;
            }
            QLabel#insightCardTitle {
                color: #6b6b67;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0;
            }
            QLabel#insightCardValue {
                color: #202123;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#insightCardMeta {
                color: #6b6b67;
                line-height: 1.45;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #ececea;
                border-radius: 8px;
                font-weight: 600;
                margin-top: 14px;
                padding-top: 12px;
            }
            QGroupBox::title {
                left: 16px;
                padding: 0 6px;
                color: #5f5f5b;
                font-size: 12px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QComboBox {
                background: #f7f7f5;
                border: 1px solid #deded9;
                border-radius: 8px;
                padding: 8px 10px;
                selection-background-color: #10a37f;
                selection-color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QComboBox:focus {
                background: #ffffff;
                border: 1px solid #10a37f;
            }
            QLabel#pathFieldLabel {
                color: #4f4f4b;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0;
            }
            QLabel#actionHintLabel {
                color: #6b6b67;
                line-height: 1.5;
            }
            QCheckBox {
                color: #202123;
                spacing: 8px;
            }
            QCheckBox#compactToggle {
                padding: 0;
                margin: 0;
                spacing: 0;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid #d5d5d0;
                background: #ffffff;
            }
            QCheckBox::indicator:hover {
                border-color: #10a37f;
            }
            QCheckBox::indicator:checked {
                background: #10a37f;
                border-color: #10a37f;
            }
            QPushButton {
                background: #10a37f;
                border: 1px solid #10a37f;
                border-radius: 8px;
                color: #ffffff;
                font-weight: 700;
                min-height: 38px;
                padding: 0 18px;
            }
            QPushButton:hover {
                background: #0e906f;
                border-color: #0e906f;
            }
            QPushButton:pressed {
                background: #0b7f62;
                border-color: #0b7f62;
            }
            QPushButton:disabled {
                background: #d9d9d4;
                border-color: #d9d9d4;
                color: #f8f8f6;
            }
            QPushButton#secondaryButton {
                background: #ffffff;
                border: 1px solid #deded9;
                color: #202123;
            }
            QPushButton#secondaryButton:hover {
                background: #f7f7f5;
                border-color: #c8c8c2;
            }
            QPushButton#dangerButton {
                background: #b42318;
                border-color: #b42318;
            }
            QPushButton#dangerButton:hover {
                background: #9f1f17;
                border-color: #9f1f17;
            }
            QProgressBar {
                background: #f0f0ed;
                border: 1px solid #e0e0db;
                border-radius: 8px;
                min-height: 24px;
                text-align: center;
                font-weight: 700;
                color: #3f3f3b;
            }
            QProgressBar::chunk {
                background: #10a37f;
                border-radius: 7px;
            }
            QFrame#previewPane {
                background: #ffffff;
                border: 1px solid #ececea;
                border-radius: 8px;
            }
            QFrame#primaryPreviewPane {
                background: #ffffff;
                border: 1px solid #e6e6e1;
                border-radius: 8px;
            }
            QLabel#previewTitle {
                color: #202123;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#previewImage {
                background: #f5f7f1;
                border: 1px dashed #d9d9d2;
                border-radius: 8px;
                color: #73736f;
                padding: 16px;
            }
            QLabel#heroPreviewImage {
                background: #f7f7f5;
                border: 1px dashed #d6d6cf;
                border-radius: 8px;
                color: #6b6b67;
                padding: 18px;
            }
            QLabel#previewCaption {
                color: #73736f;
            }
            QLabel#heroPreviewCaption {
                color: #5f5f5b;
            }
            QLabel#environmentLabel {
                border-radius: 8px;
                font-weight: 600;
                padding: 13px 15px;
            }
            QLabel#statusLabel {
                color: #6b6b67;
                line-height: 1.45;
            }
            QLabel#planDetails {
                background: #f7f7f5;
                border: 1px solid #e3e3de;
                border-radius: 8px;
                color: #4f4f4b;
                font-family: "Cascadia Code", "JetBrains Mono", "Microsoft YaHei UI";
                line-height: 1.5;
                padding: 12px 14px;
            }
            QTextEdit#summaryBox {
                background: #202123;
                border: 1px solid #343541;
                border-radius: 8px;
                color: #f7f7f5;
                font-family: "Cascadia Code", "JetBrains Mono", "Microsoft YaHei UI";
                line-height: 1.55;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 4px 0 4px 0;
            }
            QScrollBar::handle:vertical {
                background: #d2d2cc;
                border-radius: 5px;
                min-height: 36px;
            }
            QScrollBar::handle:vertical:hover {
                background: #bdbdb7;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
                width: 0;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 10px;
                margin: 0 4px 0 4px;
            }
            QScrollBar::handle:horizontal {
                background: #d2d2cc;
                border-radius: 5px;
                min-width: 36px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                height: 0;
                width: 0;
            }
            """
        )

    def _build_ui(self) -> None:
        """Create the window layout and all major interface sections."""
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(0)

        shell = QFrame()
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(14, 14, 14, 14)
        shell_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 910])

        shell_layout.addWidget(self._build_header_card())
        shell_layout.addWidget(splitter, 1)
        root_layout.addWidget(shell)
        self.setCentralWidget(central)

    def _build_header_card(self) -> QFrame:
        """Build a compact app bar with calm, tool-oriented framing."""
        card = QFrame()
        card.setObjectName("topBar")
        layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, card)
        self._top_bar_layout = layout
        layout.setContentsMargins(8, 8, 8, 10)
        layout.setSpacing(16)

        title_widget = QWidget()
        title_widget.setMinimumWidth(0)
        title_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        title_column = QVBoxLayout(title_widget)
        title_column.setSpacing(6)
        title_column.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("风格迁移工作台")
        self.top_bar_title_label = title_label
        title_label.setObjectName("topBarTitle")
        title_column.addWidget(title_label)

        layout.addWidget(title_widget, 1)
        return card

    def _build_control_panel(self) -> QWidget:
        """Build the left-side form panel with inputs and actions."""
        panel_container = QFrame()
        panel_container.setObjectName("sidebarSurface")
        panel_container.setMinimumWidth(0)
        panel_container.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        panel_layout = QVBoxLayout(panel_container)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(10)

        panel_layout.addWidget(self._build_environment_group())
        panel_layout.addWidget(self._build_input_group())
        panel_layout.addWidget(self._build_parameter_group())
        panel_layout.addWidget(self._build_action_group())
        panel_layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("sidebarScroll")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setWidget(panel_container)
        self.sidebar_scroll = scroll_area
        return scroll_area

    def _build_workspace_header(self) -> QFrame:
        """Build a compact header for the preview workspace."""
        card = QFrame()
        card.setObjectName("workspaceHeader")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        title = QLabel("预览与结果")
        self.workspace_title_label = title
        title.setObjectName("workspaceTitle")
        layout.addWidget(title, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return card

    def _build_preview_panel(self) -> QWidget:
        """Build the right-side preview and output summary area."""
        container = QFrame()
        container.setObjectName("workspaceSurface")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_workspace_header())
        layout.addWidget(self._build_live_summary_group())

        self.content_preview = PreviewPane(
            "内容参考",
            "选择内容图后，这里会显示参考缩略图。",
        )
        self.content_preview.configure_density(
            image_min_size=QSize(148, 78),
            min_height=132,
            margins=(12, 12, 12, 12),
            spacing=8,
        )
        self.style_preview = PreviewPane(
            "风格参考",
            "选择风格图后，这里会显示参考缩略图。",
        )
        self.style_preview.configure_density(
            image_min_size=QSize(148, 78),
            min_height=132,
            margins=(12, 12, 12, 12),
            spacing=8,
        )
        self.result_preview = PreviewPane(
            "结果画布",
            "生成完成后，这里会显示最终输出图像。",
        )
        self.result_preview.setObjectName("primaryPreviewPane")
        self.result_preview.image_label.setObjectName("heroPreviewImage")
        self.result_preview.caption_label.setObjectName("heroPreviewCaption")
        self.result_preview.configure_density(
            image_min_size=QSize(300, 220),
            min_height=340,
            margins=(16, 16, 16, 16),
            spacing=10,
        )

        source_column = QWidget()
        self.source_column = source_column
        source_column.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        source_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, source_column)
        self.source_preview_layout = source_layout
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(12)
        source_layout.addWidget(self.content_preview, 1)
        source_layout.addWidget(self.style_preview, 1)
        source_column.setMinimumWidth(280)
        source_column.setMaximumWidth(340)
        source_column.setMinimumHeight(276)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview_splitter = top_splitter
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setMinimumHeight(320)
        top_splitter.addWidget(self.result_preview)
        top_splitter.addWidget(source_column)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 0)
        top_splitter.setSizes([760, 300])

        summary_group = QGroupBox("运行记录")
        summary_layout = QVBoxLayout(summary_group)
        self.output_summary = QTextEdit()
        self.output_summary.setObjectName("summaryBox")
        self.output_summary.setReadOnly(True)
        self.output_summary.setMinimumHeight(190)
        self.output_summary.setPlainText(
            "这里会显示输出路径、参数记录、损失信息和本次运行摘要。"
        )
        summary_layout.addWidget(self.output_summary)

        layout.addWidget(top_splitter)
        layout.addWidget(summary_group)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("workspaceScroll")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setWidget(container)
        self.workspace_scroll = scroll_area
        return scroll_area

    def _build_environment_group(self) -> QGroupBox:
        """Build the environment-status section."""
        group = QGroupBox("设备状态")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self.environment_label = QLabel()
        self.environment_label.setWordWrap(True)
        self.environment_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.environment_label.setObjectName("environmentLabel")

        self.status_label = QLabel("选择内容图和风格图后即可开始生成。")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("待开始")

        layout.addWidget(self.environment_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        return group

    def _build_input_group(self) -> QGroupBox:
        """Build file path controls for content, style, mask, and output."""
        group = QGroupBox("素材与输出")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        self.content_input, self.content_browse_button = self._create_path_row(
            layout,
            "内容图",
            placeholder_text="例如：examples\\content.jpg",
        )
        self.style_input, self.style_browse_button = self._create_path_row(
            layout,
            "风格图",
            placeholder_text="例如：examples\\style.jpg",
        )
        self.mask_input, self.mask_browse_button = self._create_path_row(
            layout,
            "遮罩图（可选）",
            include_clear_button=True,
            placeholder_text="局部风格迁移时可选，用于限定生效区域。",
        )
        self.output_input, self.output_browse_button = self._create_path_row(
            layout,
            "输出图",
            save_dialog=True,
            placeholder_text="默认输出到 outputs\\result.png，并同步生成 JSON 参数文件。",
        )
        self.output_input.setText(str(default_output_path()))
        return group

    def _build_parameter_group(self) -> QGroupBox:
        """Build parameter widgets for the NST run settings."""
        group = QGroupBox("生成参数")
        form = QFormLayout(group)
        form.setContentsMargins(16, 18, 16, 14)
        form.setSpacing(12)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(MIN_NUM_STEPS, MAX_NUM_STEPS)
        self.steps_spin.setSingleStep(50)
        self.steps_spin.setValue(DEFAULT_NUM_STEPS)

        self.style_strength_spin = QDoubleSpinBox()
        self.style_strength_spin.setRange(MIN_STYLE_STRENGTH, MAX_STYLE_STRENGTH)
        self.style_strength_spin.setSingleStep(0.1)
        self.style_strength_spin.setDecimals(2)
        self.style_strength_spin.setValue(DEFAULT_STYLE_STRENGTH)

        self.content_blend_spin = QDoubleSpinBox()
        self.content_blend_spin.setRange(MIN_CONTENT_BLEND, MAX_CONTENT_BLEND)
        self.content_blend_spin.setSingleStep(0.05)
        self.content_blend_spin.setDecimals(2)
        self.content_blend_spin.setValue(DEFAULT_CONTENT_BLEND)
        self.content_blend_spin.setToolTip("数值越高，结果越接近原图。")

        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(GUI_MIN_IMAGE_SIZE, GUI_MAX_IMAGE_SIZE)
        self.image_size_spin.setSingleStep(64)
        self.image_size_spin.setValue(DEFAULT_IMAGE_SIZE)

        self.backbone_combo = QComboBox()
        for backbone_name, label in BACKBONE_LABELS.items():
            self.backbone_combo.addItem(label, backbone_name)
        self.backbone_combo.setCurrentIndex(
            self.backbone_combo.findData(DEFAULT_BACKBONE)
        )
        self.backbone_combo.setToolTip(
            "VGG19 用于论文复现基线，ResNet50 用于扩展对比实验。"
        )

        self.keep_color_checkbox = self._create_compact_toggle(
            "保留内容图像原有色彩，仅迁移纹理与笔触。"
        )
        self.histogram_loss_checkbox = self._create_compact_toggle(
            "增加 activation histogram matching，减少伪影和脏纹理。"
        )
        self.histogram_loss_checkbox.setToolTip(
            "增加 activation histogram matching，减少伪影和脏纹理。"
        )
        self.enhanced_mode_checkbox = self._create_compact_toggle(
            "使用平均池化、内容加噪声初始化和更激进的默认参数，强化风格表达。"
        )
        self.enhanced_mode_checkbox.setToolTip(
            "使用平均池化、内容加噪声初始化和更激进的默认参数，强化风格表达。"
        )
        self.enhanced_preset_button = QPushButton("强化预设")
        self.enhanced_preset_button.setToolTip("一键填入强化模式推荐参数。")
        self.enhanced_preset_button.setObjectName("secondaryButton")
        self.reset_parameters_button = QPushButton("恢复默认")
        self.reset_parameters_button.setToolTip("恢复到项目默认参数组合。")
        self.reset_parameters_button.setObjectName("secondaryButton")

        preset_actions = QWidget()
        preset_actions.setObjectName("presetPanel")
        preset_layout = QHBoxLayout(preset_actions)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(8)
        preset_layout.addWidget(self.enhanced_preset_button)
        preset_layout.addWidget(self.reset_parameters_button)
        preset_layout.addStretch(1)

        form.addRow("优化步数", self.steps_spin)
        form.addRow("风格强度", self.style_strength_spin)
        form.addRow("原图保留度", self.content_blend_spin)
        form.addRow("图像尺寸", self.image_size_spin)
        form.addRow("特征骨干", self.backbone_combo)
        form.addRow("保留色彩", self.keep_color_checkbox)
        form.addRow("直方图约束", self.histogram_loss_checkbox)
        form.addRow("强化模式", self.enhanced_mode_checkbox)
        form.addRow("", preset_actions)
        return group

    def _create_compact_toggle(self, tooltip: str = "") -> QCheckBox:
        """Create a label-free toggle control for compact form rows."""
        checkbox = QCheckBox()
        checkbox.setObjectName("compactToggle")
        checkbox.setText("")
        checkbox.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        if tooltip:
            checkbox.setToolTip(tooltip)
        return checkbox

    def _build_live_summary_group(self) -> QGroupBox:
        """Build a compact live summary for the current generation plan."""
        group = QGroupBox("当前方案")
        self.live_summary_group = group
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        cards_layout = QGridLayout()
        self.summary_cards_layout = cards_layout
        cards_layout.setHorizontalSpacing(10)
        cards_layout.setVerticalSpacing(10)

        mode_card, self.mode_value_label, self.mode_meta_label = self._create_insight_card("模式")
        size_card, self.size_value_label, self.size_meta_label = self._create_insight_card("输出规格")
        balance_card, self.balance_value_label, self.balance_meta_label = self._create_insight_card("风格倾向")
        control_card, self.control_value_label, self.control_meta_label = self._create_insight_card("附加控制")
        self.summary_cards = [
            mode_card,
            size_card,
            balance_card,
            control_card,
        ]

        self._apply_summary_card_layout(columns=2)

        self.plan_details_label = QLabel()
        self.plan_details_label.setObjectName("planDetails")
        self.plan_details_label.setWordWrap(True)
        self.plan_details_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        layout.addLayout(cards_layout)
        layout.addWidget(self.plan_details_label)
        return group

    def _build_action_group(self) -> QGroupBox:
        """Build the primary run and cancel action area."""
        group = QGroupBox("执行控制")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        button_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.action_button_layout = button_row
        self.run_button = QPushButton("开始风格迁移")
        self.cancel_button = QPushButton("取消当前任务")
        self.cancel_button.setObjectName("dangerButton")
        self.cancel_button.setEnabled(False)

        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)

        self.action_hint_label = QLabel(
            "任务会在后台线程执行，界面保持响应。完成后会同时保存输出图像和 JSON 参数记录。"
        )
        self.action_hint_label.setObjectName("actionHintLabel")
        self.action_hint_label.setWordWrap(True)
        self.action_hint_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        layout.addLayout(button_row)
        layout.addWidget(self.action_hint_label)
        return group

    def _create_insight_card(self, title: str) -> tuple[QFrame, QLabel, QLabel]:
        """Create a small card used by the live plan summary."""
        card = QFrame()
        card.setObjectName("insightCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("insightCardTitle")
        value_label = QLabel()
        value_label.setObjectName("insightCardValue")
        value_label.setWordWrap(True)
        value_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        meta_label = QLabel()
        meta_label.setObjectName("insightCardMeta")
        meta_label.setWordWrap(True)
        meta_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(meta_label)
        layout.addStretch(1)
        return card, value_label, meta_label

    def _create_path_row(
        self,
        parent_layout: QVBoxLayout,
        label_text: str,
        *,
        include_clear_button: bool = False,
        save_dialog: bool = False,
        placeholder_text: str = "",
    ) -> tuple[QLineEdit, QPushButton]:
        """Create a labeled file-picker row."""
        label = QLabel(label_text)
        label.setObjectName("pathFieldLabel")

        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder_text)
        line_edit.setClearButtonEnabled(True)
        line_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        browse_button = QPushButton("选择")
        browse_button.setObjectName("secondaryButton")
        browse_button.setProperty("save_dialog", save_dialog)
        browse_button.setMinimumWidth(64)

        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse_button)

        if include_clear_button:
            clear_button = QPushButton("清空")
            clear_button.setObjectName("secondaryButton")
            clear_button.setMinimumWidth(64)
            clear_button.clicked.connect(line_edit.clear)
            row_layout.addWidget(clear_button)

        parent_layout.addWidget(label)
        parent_layout.addLayout(row_layout)
        return line_edit, browse_button

    def _bind_events(self) -> None:
        """Connect interactive controls to their current UI behavior."""
        self.content_browse_button.clicked.connect(
            lambda: self._pick_image_file(self.content_input)
        )
        self.style_browse_button.clicked.connect(
            lambda: self._pick_image_file(self.style_input)
        )
        self.mask_browse_button.clicked.connect(
            lambda: self._pick_image_file(self.mask_input)
        )
        self.output_browse_button.clicked.connect(self._pick_output_file)
        self.enhanced_preset_button.clicked.connect(self._apply_enhanced_preset)
        self.reset_parameters_button.clicked.connect(self._reset_parameters_to_default)
        self.run_button.clicked.connect(self._start_run)
        self.cancel_button.clicked.connect(self._request_cancel)
        self.content_input.editingFinished.connect(self._refresh_source_previews)
        self.style_input.editingFinished.connect(self._refresh_source_previews)
        self.content_input.textChanged.connect(self._refresh_live_summary)
        self.style_input.textChanged.connect(self._refresh_live_summary)
        self.mask_input.textChanged.connect(self._refresh_live_summary)
        self.output_input.textChanged.connect(self._refresh_live_summary)
        self.steps_spin.valueChanged.connect(self._refresh_live_summary)
        self.style_strength_spin.valueChanged.connect(self._refresh_live_summary)
        self.content_blend_spin.valueChanged.connect(self._refresh_live_summary)
        self.image_size_spin.valueChanged.connect(self._refresh_live_summary)
        self.backbone_combo.currentIndexChanged.connect(self._refresh_live_summary)
        self.keep_color_checkbox.toggled.connect(self._refresh_live_summary)
        self.histogram_loss_checkbox.toggled.connect(self._refresh_live_summary)
        self.enhanced_mode_checkbox.toggled.connect(self._refresh_live_summary)
        self.run_button.setEnabled(self._cuda_ready)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep the tool layout stable when the window width changes."""
        super().resizeEvent(event)
        self._sync_responsive_layouts()

    def _sync_responsive_layouts(self) -> None:
        """Adjust layout directions and grids for narrower window widths."""
        if hasattr(self, "_top_bar_layout"):
            top_bar_direction = QBoxLayout.Direction.LeftToRight
            if self._top_bar_layout.direction() != top_bar_direction:
                self._top_bar_layout.setDirection(top_bar_direction)

        if hasattr(self, "action_button_layout") and hasattr(self, "sidebar_scroll"):
            sidebar_width = self.sidebar_scroll.viewport().width()
            action_direction = (
                QBoxLayout.Direction.TopToBottom
                if sidebar_width < 360
                else QBoxLayout.Direction.LeftToRight
            )
            if self.action_button_layout.direction() != action_direction:
                self.action_button_layout.setDirection(action_direction)

        if hasattr(self, "preview_splitter") and hasattr(self, "source_column"):
            preview_orientation = Qt.Orientation.Horizontal
            if self.preview_splitter.orientation() != preview_orientation:
                self.preview_splitter.setOrientation(preview_orientation)
                if preview_orientation == Qt.Orientation.Vertical:
                    self.source_preview_layout.setDirection(
                        QBoxLayout.Direction.LeftToRight
                    )
                    self.source_column.setMinimumWidth(0)
                    self.source_column.setMaximumWidth(16777215)
                    self.preview_splitter.setSizes([440, 260])
                else:
                    self.source_preview_layout.setDirection(
                        QBoxLayout.Direction.TopToBottom
                    )
                    self.source_column.setMinimumWidth(240)
                    self.source_column.setMaximumWidth(340)
                    self.preview_splitter.setSizes([760, 300])

        if hasattr(self, "summary_cards"):
            summary_group_width = self.live_summary_group.width() if hasattr(self, "live_summary_group") else self.width()
            if summary_group_width >= 760:
                summary_columns = 4
            elif summary_group_width >= 480:
                summary_columns = 2
            else:
                summary_columns = 1
            self._apply_summary_card_layout(columns=summary_columns)
            self._refresh_summary_cards_geometry()

        for label in (
            getattr(self, "top_bar_subtitle_label", None),
            getattr(self, "top_bar_meta_label", None),
            getattr(self, "workspace_subtitle_label", None),
            getattr(self, "environment_label", None),
            getattr(self, "status_label", None),
            getattr(self, "plan_details_label", None),
            getattr(self, "action_hint_label", None),
        ):
            if label is not None:
                self._fit_wrapped_label_height(label, extra_padding=4)

    def _apply_summary_card_layout(self, columns: int) -> None:
        """Place summary cards in one or two columns depending on width."""
        if not hasattr(self, "summary_cards_layout") or not hasattr(self, "summary_cards"):
            return
        if getattr(self, "_summary_columns", None) == columns:
            return

        while self.summary_cards_layout.count():
            item = self.summary_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for index, card in enumerate(self.summary_cards):
            row = index // columns
            col = index % columns
            self.summary_cards_layout.addWidget(card, row, col)

        for col in range(len(self.summary_cards)):
            self.summary_cards_layout.setColumnStretch(col, 1 if col < columns else 0)

        self._summary_columns = columns

    def _refresh_summary_cards_geometry(self) -> None:
        """Keep plan summary cards readable after text or width changes."""
        if not hasattr(self, "summary_cards"):
            return

        card_specs = [
            (self.summary_cards[0], self.mode_value_label, self.mode_meta_label),
            (self.summary_cards[1], self.size_value_label, self.size_meta_label),
            (self.summary_cards[2], self.balance_value_label, self.balance_meta_label),
            (self.summary_cards[3], self.control_value_label, self.control_meta_label),
        ]
        for card, value_label, meta_label in card_specs:
            self._fit_wrapped_label_height(value_label, extra_padding=2)
            self._fit_wrapped_label_height(meta_label, extra_padding=2)
            card.setMinimumHeight(max(84, card.layout().sizeHint().height() + 6))
            card.updateGeometry()
        if hasattr(self, "live_summary_group"):
            content_height = (
                self.summary_cards_layout.sizeHint().height()
                + self.plan_details_label.height()
                + 62
            )
            self.live_summary_group.setMinimumHeight(max(182, content_height))

    def _fit_wrapped_label_height(
        self,
        label: QLabel,
        *,
        extra_padding: int = 0,
    ) -> None:
        """Resize a wrapped label vertically so text is never clipped."""
        if not label.wordWrap():
            return

        available_width = label.contentsRect().width() or label.width()
        if available_width <= 0:
            available_width = max(label.sizeHint().width(), 120)

        text_rect = label.fontMetrics().boundingRect(
            0,
            0,
            available_width,
            4000,
            Qt.TextFlag.TextWordWrap,
            label.text(),
        )
        label.setFixedHeight(text_rect.height() + extra_padding)
        label.updateGeometry()

    def _pick_image_file(self, target_input: QLineEdit) -> None:
        """Open an image file picker and store the selected path."""
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像文件",
            str(Path.cwd()),
            GUI_IMAGE_FILE_FILTER,
        )
        if selected_path:
            target_input.setText(selected_path)
            self._refresh_source_previews()

    def _pick_output_file(self) -> None:
        """Open a save dialog for the generated output image path."""
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择输出图像路径",
            self.output_input.text().strip() or str(default_output_path()),
            GUI_IMAGE_FILE_FILTER,
        )
        if selected_path:
            self.output_input.setText(selected_path)

    def _apply_enhanced_preset(self) -> None:
        """Apply the stronger preset tuned for the enhanced mode."""
        self.enhanced_mode_checkbox.setChecked(True)
        self.steps_spin.setValue(max(self.steps_spin.value(), ENHANCED_MODE_NUM_STEPS))
        self.style_strength_spin.setValue(ENHANCED_MODE_STYLE_STRENGTH)
        self.content_blend_spin.setValue(ENHANCED_MODE_CONTENT_BLEND)
        self.keep_color_checkbox.setChecked(False)
        self.histogram_loss_checkbox.setChecked(False)

    def _current_backbone(self) -> str:
        """Return the currently selected style-transfer backbone name."""
        return str(self.backbone_combo.currentData() or DEFAULT_BACKBONE)

    def _current_backbone_short_label(self) -> str:
        """Return a compact label for the selected backbone."""
        return "VGG19" if self._current_backbone() == DEFAULT_BACKBONE else "ResNet50"

    def _reset_parameters_to_default(self) -> None:
        """Restore the editable generation parameters to the default profile."""
        self.steps_spin.setValue(DEFAULT_NUM_STEPS)
        self.style_strength_spin.setValue(DEFAULT_STYLE_STRENGTH)
        self.content_blend_spin.setValue(DEFAULT_CONTENT_BLEND)
        self.image_size_spin.setValue(DEFAULT_IMAGE_SIZE)
        self.backbone_combo.setCurrentIndex(self.backbone_combo.findData(DEFAULT_BACKBONE))
        self.keep_color_checkbox.setChecked(False)
        self.histogram_loss_checkbox.setChecked(False)
        self.enhanced_mode_checkbox.setChecked(False)

    def _refresh_live_summary(self, *_args: object) -> None:
        """Refresh the live plan cards and execution hint from current form values."""
        enhanced_mode = self.enhanced_mode_checkbox.isChecked()
        keep_color = self.keep_color_checkbox.isChecked()
        histogram_loss = self.histogram_loss_checkbox.isChecked()
        has_mask = bool(self.mask_input.text().strip())
        image_size = self.image_size_spin.value()
        style_strength = self.style_strength_spin.value()
        content_blend = self.content_blend_spin.value()
        backbone_name = self._current_backbone()
        backbone_label = self.backbone_combo.currentText().strip()
        backbone_short_label = self._current_backbone_short_label()

        output_path = Path(self.output_input.text().strip() or default_output_path()).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")
        metadata_path = metadata_output_path(output_path)

        if enhanced_mode:
            mode_value = "强化模式"
            mode_meta = "更强纹理、更少原图约束，适合追求明显风格表现。"
        else:
            mode_value = "标准模式"
            mode_meta = "均衡配置，适合常规生成与快速冒烟验证。"

        self.mode_value_label.setText(mode_value)
        self.mode_meta_label.setText(f"{mode_meta} 当前骨干：{backbone_label}。")

        self.size_value_label.setText(f"{image_size} px")
        self.size_meta_label.setText("按最长边缩放，结果图会保持原始宽高比。")

        self.balance_value_label.setText(
            f"风格 {style_strength:.2f} / 保留 {content_blend:.2f}"
        )
        self.balance_meta_label.setText(
            self._describe_style_profile(
                style_strength,
                content_blend,
                enhanced_mode,
                histogram_loss,
            )
        )

        self.control_value_label.setText(
            " | ".join(
                [
                    f"骨干 {backbone_short_label}",
                    f"保色 {'开' if keep_color else '关'}",
                    f"遮罩 {'开' if has_mask else '关'}",
                    f"直方图 {'开' if histogram_loss else '关'}",
                ]
            )
        )
        self.control_meta_label.setText("输出时会同时保存 PNG 图像与 JSON 参数记录。")

        self.plan_details_label.setText(
            "\n".join(
                [
                    f"内容：{self._format_path_for_display(self.content_input.text().strip(), '未选择内容图')}",
                    f"风格：{self._format_path_for_display(self.style_input.text().strip(), '未选择风格图')}",
                    f"骨干：{backbone_name}",
                    f"输出：{self._format_path_for_display(str(output_path), 'outputs/result.png')}",
                    f"记录：{self._format_path_for_display(str(metadata_path), metadata_path.name)}",
                ]
            )
        )

        hint_bits = [
            mode_value,
            backbone_short_label,
            f"最长边 {image_size}px",
            "保色开启" if keep_color else "允许改色",
            "局部遮罩" if has_mask else "全局作用",
        ]
        if histogram_loss:
            hint_bits.append("直方图稳定")
        self.action_hint_label.setText(
            "当前方案："
            + " / ".join(hint_bits)
            + "。任务会在后台线程执行，完成后自动保存图像与 JSON 参数记录。"
        )
        self._sync_responsive_layouts()

    def _describe_style_profile(
        self,
        style_strength: float,
        content_blend: float,
        enhanced_mode: bool,
        histogram_loss: bool,
    ) -> str:
        """Describe the current style profile in a short user-facing sentence."""
        if histogram_loss:
            return "当前会额外约束激活分布，通常能减少脏纹理和局部伪影。"
        if enhanced_mode:
            return "笔触更重、纹理更明显，适合追求接近论文示例的强风格化结果。"
        if style_strength >= 2.0 or content_blend <= 0.1:
            return "当前偏强风格，结构保留较少，结果会更远离原图。"
        if content_blend >= 0.45:
            return "当前偏轻风格，结果会更贴近原图结构与颜色。"
        return "当前为均衡风格，适合大多数常规演示场景。"

    def _format_path_for_display(self, raw_path: str, fallback: str) -> str:
        """Format a filesystem path for compact display inside the live summary."""
        if not raw_path:
            return fallback

        path = Path(raw_path).expanduser()
        if path.is_absolute():
            try:
                return str(path.relative_to(Path.cwd()))
            except ValueError:
                return path.name if len(str(path)) > 56 else str(path)
        return str(path)

    def _refresh_environment_status(self) -> None:
        """Update the environment banner and current run availability."""
        status_text = build_startup_status_message()
        if self._cuda_ready:
            self.environment_label.setStyleSheet(
                "background: #e5f4eb; color: #1e6a39;"
            )
            self.environment_label.setText(
                "CUDA 环境就绪。\n"
                f"{status_text}"
            )
        else:
            self.environment_label.setStyleSheet(
                "background: #fbe8e8; color: #8a1f1f;"
            )
            self.environment_label.setText(
                "当前无法启动生成任务。\n"
                f"{status_text}"
            )
        self.run_button.setEnabled(self._cuda_ready and self._worker_thread is None)

    def _refresh_source_previews(self) -> None:
        """Load previews for the currently selected content and style images."""
        self._set_preview_from_path(
            self.content_preview,
            self.content_input.text().strip(),
        )
        self._set_preview_from_path(
            self.style_preview,
            self.style_input.text().strip(),
        )

    def _set_preview_from_path(
        self,
        preview_pane: PreviewPane,
        raw_path: str,
    ) -> None:
        """Load preview bytes from a valid path or restore the placeholder."""
        if not raw_path:
            preview_pane.clear_preview("尚未选择文件。")
            return

        path = Path(raw_path).expanduser()
        if not path.exists() or not path.is_file():
            preview_pane.clear_preview(f"未找到文件：{path}")
            return

        try:
            preview_bytes = load_preview_png_bytes(path)
        except Exception as exc:  # pragma: no cover - preview is best-effort UI behavior
            preview_pane.clear_preview(f"预览加载失败：{exc}")
            return

        preview_pane.set_preview_bytes(preview_bytes, self._format_preview_caption(path))

    def _format_preview_caption(self, path: Path) -> str:
        """Build a compact caption for a preview pane."""
        return f"{path.name}\n{path}"

    def _collect_run_request(self) -> StyleTransferRunRequest:
        """Read the current UI values and validate them into a run request."""
        content_path = validate_image_path(
            self.content_input.text().strip(),
            "内容图像",
        )
        style_path = validate_image_path(
            self.style_input.text().strip(),
            "风格图像",
        )
        mask_path = validate_optional_image_path(
            self.mask_input.text().strip(),
            "遮罩图像",
        )
        require_cuda()
        output_path = validate_output_image_path(self.output_input.text().strip())
        num_steps = validate_num_steps(self.steps_spin.value())
        style_strength = validate_style_strength(self.style_strength_spin.value())
        content_blend = validate_content_blend(self.content_blend_spin.value())
        image_size = validate_image_size(self.image_size_spin.value())

        return StyleTransferRunRequest(
            content_path=content_path,
            style_path=style_path,
            mask_path=mask_path,
            output_path=output_path,
            num_steps=num_steps,
            style_strength=style_strength,
            content_blend=content_blend,
            image_size=image_size,
            keep_color=self.keep_color_checkbox.isChecked(),
            backbone=self._current_backbone(),
            histogram_loss=self.histogram_loss_checkbox.isChecked(),
            enhanced_mode=self.enhanced_mode_checkbox.isChecked(),
            paper_mode=False,
        )

    def _start_run(self) -> None:
        """Validate the form and launch the background worker."""
        if self._worker_thread is not None:
            return

        try:
            request = self._collect_run_request()
        except ValidationError as exc:
            self._cuda_ready = is_cuda_ready()
            self._refresh_environment_status()
            self.output_summary.setPlainText(str(exc))
            self._show_error("配置无效", str(exc))
            return

        self.result_preview.clear_preview("当前任务执行中，生成完成后会显示新结果。")
        self._set_running_state(True)
        self._set_status("正在启动后台任务...", 0, "准备中")

        self._worker_thread = QThread(self)
        self._worker = StyleTransferWorker(request)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._handle_worker_progress)
        self._worker.succeeded.connect(self._handle_worker_success)
        self._worker.failed.connect(self._handle_worker_failure)
        self._worker.cancelled.connect(self._handle_worker_cancelled)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._handle_worker_finished)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self.cancel_requested.connect(self._worker.cancel)

        self._worker_thread.start()

    def _request_cancel(self) -> None:
        """Forward a cooperative cancellation request to the worker thread."""
        if self._worker is None:
            return
        self.cancel_button.setEnabled(False)
        self._set_status("已请求取消，正在等待安全检查点...", self.progress_bar.value())
        self.cancel_requested.emit()

    def _handle_worker_progress(self, progress: StyleTransferRunProgress) -> None:
        """Reflect worker progress updates in the GUI."""
        message = progress.message
        if progress.content_loss is not None and progress.style_loss is not None:
            message = (
                f"{progress.message}\n"
                f"内容损失：{progress.content_loss:.4f} | "
                f"风格损失：{progress.style_loss:.4f}"
            )
        self._set_status(message, progress.percent)

    def _handle_worker_success(self, result: StyleTransferRunResult) -> None:
        """Handle a completed background NST run."""
        self.result_preview.set_preview_bytes(
            result.preview_png_bytes,
            self._format_preview_caption(result.output_image_path),
        )
        self.output_summary.setPlainText(
            "\n".join(
                [
                    result.metadata_summary,
                    f"内容损失：{result.content_loss:.6f}",
                    f"风格损失：{result.style_loss:.6f}",
                    f"直方图损失：{result.histogram_loss:.6f}",
                    f"应用遮罩：{'是' if result.applied_mask else '否'}",
                ]
            )
        )
        self._set_status("生成完成，结果与参数记录已写入磁盘。", 100, "已完成")
        self._set_running_state(False)

    def _handle_worker_failure(self, message: str) -> None:
        """Handle a background worker failure."""
        self.result_preview.clear_preview("本次任务执行失败，未生成新的结果图像。")
        self.output_summary.setPlainText(message)
        self._show_error("生成失败", message)
        self._set_status("生成失败，请检查输入路径、参数和 CUDA 环境。", 0, "失败")
        self._set_running_state(False)

    def _handle_worker_cancelled(self, message: str) -> None:
        """Handle a cooperatively cancelled NST run."""
        self.result_preview.clear_preview("本次任务已取消，没有生成新的结果图像。")
        self._set_status(message or "任务已取消。", 0, "已取消")
        self.output_summary.setPlainText("当前任务在完成前已被取消。")
        self._set_running_state(False)

    def _handle_worker_finished(self) -> None:
        """Keep the main window responsive once the worker exits."""
        self.cancel_button.setEnabled(False)

    def _cleanup_worker(self) -> None:
        """Drop thread and worker references after the Qt thread exits."""
        if self._worker is not None:
            try:
                self.cancel_requested.disconnect(self._worker.cancel)
            except (RuntimeError, TypeError):
                pass
        self._worker = None
        self._worker_thread = None
        self._refresh_environment_status()

    def _set_running_state(self, is_running: bool) -> None:
        """Toggle the form controls between idle and running states."""
        for widget in (
            self.content_input,
            self.style_input,
            self.mask_input,
            self.output_input,
            self.content_browse_button,
            self.style_browse_button,
            self.mask_browse_button,
            self.output_browse_button,
            self.steps_spin,
            self.style_strength_spin,
            self.content_blend_spin,
            self.image_size_spin,
            self.backbone_combo,
            self.keep_color_checkbox,
            self.histogram_loss_checkbox,
            self.enhanced_mode_checkbox,
            self.enhanced_preset_button,
            self.reset_parameters_button,
        ):
            widget.setEnabled(not is_running)

        self.run_button.setEnabled((not is_running) and self._cuda_ready)
        self.cancel_button.setEnabled(is_running)

    def _set_status(self, message: str, progress_value: int, progress_text: str | None = None) -> None:
        """Update the status label and progress bar together."""
        self.status_label.setText(message)
        self.progress_bar.setValue(max(0, min(100, progress_value)))
        self.progress_bar.setFormat(progress_text or f"{self.progress_bar.value()}%")

    def _show_error(self, title: str, message: str) -> None:
        """Show a blocking error dialog with the provided message."""
        QMessageBox.critical(self, title, message)


def main() -> int:
    """Start the Qt event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName(APP_ORGANIZATION)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
