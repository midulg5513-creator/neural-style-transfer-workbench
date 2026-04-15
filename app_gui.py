"""Desktop entry point for the PySide6 application."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_STEPS,
    DEFAULT_STYLE_STRENGTH,
    GUI_IMAGE_FILE_FILTER,
    GUI_MAX_IMAGE_SIZE,
    GUI_MIN_IMAGE_SIZE,
    GUI_PREVIEW_HEIGHT,
    GUI_PREVIEW_WIDTH,
    MAX_NUM_STEPS,
    MAX_STYLE_STRENGTH,
    MIN_NUM_STEPS,
    MIN_STYLE_STRENGTH,
)
from neural_style.utils import default_output_path, load_preview_png_bytes
from neural_style.validation import (
    ValidationError,
    build_startup_status_message,
    is_cuda_ready,
    require_cuda,
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("previewTitle")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumSize(QSize(GUI_PREVIEW_WIDTH, GUI_PREVIEW_HEIGHT))
        self.image_label.setObjectName("previewImage")

        self.caption_label = QLabel()
        self.caption_label.setWordWrap(True)
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

    def _apply_window_style(self) -> None:
        """Apply a light visual treatment for the desktop layout."""
        self.setStyleSheet(
            """
            QMainWindow {
                background: #eef3ef;
            }
            QWidget {
                color: #17324d;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC";
                font-size: 13px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QFrame#heroCard {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #17324d,
                    stop: 0.55 #1f5f66,
                    stop: 1 #2f7f67
                );
                border-radius: 24px;
                min-height: 150px;
            }
            QLabel#heroTitle {
                color: #f5fbf8;
                font-size: 29px;
                font-weight: 700;
            }
            QLabel#heroSubtitle {
                color: #d9ede8;
                font-size: 14px;
            }
            QLabel#heroBadge {
                background: rgba(255, 255, 255, 0.16);
                border-radius: 999px;
                color: #f5fbf8;
                font-size: 12px;
                font-weight: 700;
                padding: 6px 12px;
            }
            QGroupBox {
                background: #fbfdfb;
                border: 1px solid #d8e4de;
                border-radius: 16px;
                font-weight: 600;
                margin-top: 14px;
                padding-top: 10px;
            }
            QGroupBox::title {
                left: 14px;
                padding: 0 6px;
                color: #17324d;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
                background: #f7fbf9;
                border: 1px solid #c7d6ce;
                border-radius: 10px;
                padding: 7px 10px;
            }
            QPushButton {
                background: #1f6f66;
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: 700;
                min-height: 36px;
                padding: 0 16px;
            }
            QPushButton:disabled {
                background: #a9c0ba;
                color: #eef5f2;
            }
            QPushButton#secondaryButton {
                background: #d9ebe6;
                color: #164944;
            }
            QPushButton#dangerButton {
                background: #d16456;
            }
            QProgressBar {
                background: #dfebe6;
                border: 1px solid #cad8d2;
                border-radius: 10px;
                min-height: 24px;
                text-align: center;
                font-weight: 700;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #1f6f66,
                    stop: 1 #3a8a62
                );
                border-radius: 9px;
            }
            QFrame#previewPane {
                background: #ffffff;
                border: 1px solid #d8e4de;
                border-radius: 18px;
            }
            QLabel#previewTitle {
                color: #17324d;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#previewImage {
                background: #f5faf8;
                border: 1px dashed #b8cbc3;
                border-radius: 14px;
                color: #5f746d;
                padding: 14px;
            }
            QLabel#previewCaption {
                color: #4d635d;
            }
            QLabel#environmentLabel {
                border-radius: 12px;
                font-weight: 600;
                padding: 12px 14px;
            }
            QLabel#statusLabel {
                color: #3a5458;
                line-height: 1.45;
            }
            QTextEdit#summaryBox {
                background: #f8fbfa;
                line-height: 1.55;
            }
            """
        )

    def _build_ui(self) -> None:
        """Create the window layout and all major interface sections."""
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 820])

        root_layout.addWidget(self._build_header_card())
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _build_header_card(self) -> QFrame:
        """Build the top hero card for the demo-oriented interface."""
        card = QFrame()
        card.setObjectName("heroCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title_label = QLabel("神经风格迁移桌面演示界面")
        title_label.setObjectName("heroTitle")
        subtitle_label = QLabel(
            "本地 GPU 风格迁移流程，支持内容图、风格图、可选遮罩、保留原色与参数 JSON 留档。"
        )
        subtitle_label.setObjectName("heroSubtitle")
        subtitle_label.setWordWrap(True)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        for text in ("本地桌面版", "GPU 专用", "离线演示可用"):
            badge_row.addWidget(self._create_badge(text))
        badge_row.addStretch(1)

        layout.addLayout(badge_row)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return card

    def _create_badge(self, text: str) -> QLabel:
        """Create a small rounded badge used in the header."""
        badge = QLabel(text)
        badge.setObjectName("heroBadge")
        return badge

    def _build_control_panel(self) -> QWidget:
        """Build the left-side form panel with inputs and actions."""
        panel_container = QWidget()
        panel_layout = QVBoxLayout(panel_container)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(12)

        panel_layout.addWidget(self._build_environment_group())
        panel_layout.addWidget(self._build_input_group())
        panel_layout.addWidget(self._build_parameter_group())
        panel_layout.addWidget(self._build_action_group())
        panel_layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(panel_container)
        return scroll_area

    def _build_preview_panel(self) -> QWidget:
        """Build the right-side preview and output summary area."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        preview_grid = QGridLayout()
        preview_grid.setSpacing(12)
        self.content_preview = PreviewPane(
            "内容图预览",
            "选择内容图后，会在这里显示缩略预览。",
        )
        self.style_preview = PreviewPane(
            "风格图预览",
            "选择风格图后，会在这里显示缩略预览。",
        )
        self.result_preview = PreviewPane(
            "结果预览",
            "生成完成后，会在这里显示输出结果。",
        )
        preview_grid.addWidget(self.content_preview, 0, 0)
        preview_grid.addWidget(self.style_preview, 0, 1)
        preview_grid.addWidget(self.result_preview, 1, 0, 1, 2)

        summary_group = QGroupBox("结果与记录")
        summary_layout = QVBoxLayout(summary_group)
        self.output_summary = QTextEdit()
        self.output_summary.setObjectName("summaryBox")
        self.output_summary.setReadOnly(True)
        self.output_summary.setMinimumHeight(170)
        self.output_summary.setPlainText(
            "输出图像路径、参数 JSON 路径和本次运行摘要会显示在这里。"
        )
        summary_layout.addWidget(self.output_summary)

        layout.addLayout(preview_grid, 1)
        layout.addWidget(summary_group)
        return container

    def _build_environment_group(self) -> QGroupBox:
        """Build the environment-status section."""
        group = QGroupBox("运行环境")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self.environment_label = QLabel()
        self.environment_label.setWordWrap(True)
        self.environment_label.setObjectName("environmentLabel")

        self.status_label = QLabel("当前已就绪。请选择内容图和风格图，然后设置参数开始生成。")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("待命")

        layout.addWidget(self.environment_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        return group

    def _build_input_group(self) -> QGroupBox:
        """Build file path controls for content, style, mask, and output."""
        group = QGroupBox("输入与输出")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        self.content_input, self.content_browse_button = self._create_path_row(
            layout,
            "内容图像",
            placeholder_text="请选择内容图，例如：examples\\content.jpg",
        )
        self.style_input, self.style_browse_button = self._create_path_row(
            layout,
            "风格图像",
            placeholder_text="请选择风格图，例如：examples\\style.jpg",
        )
        self.mask_input, self.mask_browse_button = self._create_path_row(
            layout,
            "遮罩图像（可选）",
            include_clear_button=True,
            placeholder_text="局部风格迁移时可选，用于限定生效区域。",
        )
        self.output_input, self.output_browse_button = self._create_path_row(
            layout,
            "输出图像",
            save_dialog=True,
            placeholder_text="默认会输出到 outputs\\result.png，并同步生成 JSON 参数文件。",
        )
        self.output_input.setText(str(default_output_path()))
        return group

    def _build_parameter_group(self) -> QGroupBox:
        """Build parameter widgets for the NST run settings."""
        group = QGroupBox("参数设置")
        form = QFormLayout(group)
        form.setContentsMargins(16, 18, 16, 14)
        form.setSpacing(12)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(MIN_NUM_STEPS, MAX_NUM_STEPS)
        self.steps_spin.setSingleStep(50)
        self.steps_spin.setValue(DEFAULT_NUM_STEPS)

        self.style_strength_spin = QDoubleSpinBox()
        self.style_strength_spin.setRange(MIN_STYLE_STRENGTH, MAX_STYLE_STRENGTH)
        self.style_strength_spin.setSingleStep(0.1)
        self.style_strength_spin.setDecimals(2)
        self.style_strength_spin.setValue(DEFAULT_STYLE_STRENGTH)

        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(GUI_MIN_IMAGE_SIZE, GUI_MAX_IMAGE_SIZE)
        self.image_size_spin.setSingleStep(64)
        self.image_size_spin.setValue(DEFAULT_IMAGE_SIZE)

        self.keep_color_checkbox = QCheckBox("保留原图色彩")

        form.addRow("优化步数", self.steps_spin)
        form.addRow("风格强度", self.style_strength_spin)
        form.addRow("图像尺寸", self.image_size_spin)
        form.addRow("色彩保留", self.keep_color_checkbox)
        return group

    def _build_action_group(self) -> QGroupBox:
        """Build the primary run and cancel action area."""
        group = QGroupBox("执行控制")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("开始生成")
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setObjectName("dangerButton")
        self.cancel_button.setEnabled(False)

        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)

        self.action_hint_label = QLabel(
            "任务会在后台线程中执行，界面不会因为风格迁移而卡死。生成完成后会同时保存图像与 JSON 参数记录。"
        )
        self.action_hint_label.setWordWrap(True)
        self.action_hint_label.setStyleSheet("color: #4d635d; line-height: 1.45;")

        layout.addLayout(button_row)
        layout.addWidget(self.action_hint_label)
        return group

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
        label.setStyleSheet("font-weight: 600; color: #17324d;")

        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder_text)
        line_edit.setClearButtonEnabled(True)
        browse_button = QPushButton("选择")
        browse_button.setObjectName("secondaryButton")
        browse_button.setProperty("save_dialog", save_dialog)

        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse_button)

        if include_clear_button:
            clear_button = QPushButton("清空")
            clear_button.setObjectName("secondaryButton")
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
        self.run_button.clicked.connect(self._start_run)
        self.cancel_button.clicked.connect(self._request_cancel)
        self.content_input.editingFinished.connect(self._refresh_source_previews)
        self.style_input.editingFinished.connect(self._refresh_source_previews)
        self.run_button.setEnabled(self._cuda_ready)

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
        image_size = validate_image_size(self.image_size_spin.value())

        return StyleTransferRunRequest(
            content_path=content_path,
            style_path=style_path,
            mask_path=mask_path,
            output_path=output_path,
            num_steps=num_steps,
            style_strength=style_strength,
            image_size=image_size,
            keep_color=self.keep_color_checkbox.isChecked(),
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
            self.image_size_spin,
            self.keep_color_checkbox,
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
