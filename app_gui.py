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
        self.caption_label.setText(caption or "No file selected.")

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
            self.clear_preview("Preview data could not be decoded.")
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
                background: #f3f5f7;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7dde5;
                border-radius: 10px;
                font-weight: 600;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                left: 12px;
                padding: 0 4px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
                background: #fbfcfe;
                border: 1px solid #c8d1dc;
                border-radius: 8px;
                padding: 6px 8px;
            }
            QPushButton {
                background: #0f6cbd;
                border: none;
                border-radius: 8px;
                color: white;
                min-height: 34px;
                padding: 0 14px;
            }
            QPushButton:disabled {
                background: #9fb7cc;
                color: #eef4f8;
            }
            QPushButton#secondaryButton {
                background: #dde6f0;
                color: #17324d;
            }
            QProgressBar {
                background: #dde6f0;
                border: 1px solid #cad4df;
                border-radius: 8px;
                min-height: 24px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #0f6cbd;
                border-radius: 7px;
            }
            QFrame#previewPane {
                background: #ffffff;
                border: 1px solid #d7dde5;
                border-radius: 12px;
            }
            QLabel#previewTitle {
                color: #17324d;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#previewImage {
                background: #f6f9fc;
                border: 1px dashed #b5c4d4;
                border-radius: 10px;
                color: #607387;
                padding: 10px;
            }
            QLabel#previewCaption {
                color: #425466;
            }
            QLabel#environmentLabel {
                border-radius: 8px;
                padding: 10px 12px;
            }
            """
        )

    def _build_ui(self) -> None:
        """Create the window layout and all major interface sections."""
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        header_layout = QVBoxLayout()
        title_label = QLabel("Neural Style Transfer Desktop Studio")
        title_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #17324d;")
        subtitle_label = QLabel(
            "GPU-only local workflow with content/style selection, parameter tuning, and saved-output metadata."
        )
        subtitle_label.setStyleSheet("color: #4a6075;")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 820])

        root_layout.addLayout(header_layout)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

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
            "Content Preview",
            "The selected content image will appear here.",
        )
        self.style_preview = PreviewPane(
            "Style Preview",
            "The selected style image will appear here.",
        )
        self.result_preview = PreviewPane(
            "Result Preview",
            "The generated image preview will appear here after a successful run.",
        )
        preview_grid.addWidget(self.content_preview, 0, 0)
        preview_grid.addWidget(self.style_preview, 0, 1)
        preview_grid.addWidget(self.result_preview, 1, 0, 1, 2)

        summary_group = QGroupBox("Run Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.output_summary = QTextEdit()
        self.output_summary.setReadOnly(True)
        self.output_summary.setMinimumHeight(170)
        self.output_summary.setPlainText(
            "Saved image path, metadata path, and runtime summary will appear here."
        )
        summary_layout.addWidget(self.output_summary)

        layout.addLayout(preview_grid, 1)
        layout.addWidget(summary_group)
        return container

    def _build_environment_group(self) -> QGroupBox:
        """Build the environment-status section."""
        group = QGroupBox("Environment")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self.environment_label = QLabel()
        self.environment_label.setWordWrap(True)
        self.environment_label.setObjectName("environmentLabel")

        self.status_label = QLabel("Ready. Choose the content and style inputs to configure a run.")
        self.status_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")

        layout.addWidget(self.environment_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        return group

    def _build_input_group(self) -> QGroupBox:
        """Build file path controls for content, style, mask, and output."""
        group = QGroupBox("Inputs And Output")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        self.content_input, self.content_browse_button = self._create_path_row(
            layout,
            "Content image",
        )
        self.style_input, self.style_browse_button = self._create_path_row(
            layout,
            "Style image",
        )
        self.mask_input, self.mask_browse_button = self._create_path_row(
            layout,
            "Mask image (optional)",
            include_clear_button=True,
        )
        self.output_input, self.output_browse_button = self._create_path_row(
            layout,
            "Output image",
            save_dialog=True,
        )
        self.output_input.setText(str(default_output_path()))
        return group

    def _build_parameter_group(self) -> QGroupBox:
        """Build parameter widgets for the NST run settings."""
        group = QGroupBox("Parameters")
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

        self.keep_color_checkbox = QCheckBox("Preserve the original content colors")

        form.addRow("Optimization steps", self.steps_spin)
        form.addRow("Style strength", self.style_strength_spin)
        form.addRow("Image size", self.image_size_spin)
        form.addRow("Keep color", self.keep_color_checkbox)
        return group

    def _build_action_group(self) -> QGroupBox:
        """Build the primary run and cancel action area."""
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Start Transfer")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("secondaryButton")
        self.cancel_button.setEnabled(False)

        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)

        self.action_hint_label = QLabel(
            "Runs execute on a background worker thread so the window stays responsive."
        )
        self.action_hint_label.setWordWrap(True)
        self.action_hint_label.setStyleSheet("color: #506579;")

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
    ) -> tuple[QLineEdit, QPushButton]:
        """Create a labeled file-picker row."""
        label = QLabel(label_text)
        label.setStyleSheet("font-weight: 600; color: #17324d;")

        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        line_edit = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.setObjectName("secondaryButton")
        browse_button.setProperty("save_dialog", save_dialog)

        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse_button)

        if include_clear_button:
            clear_button = QPushButton("Clear")
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
            "Select image",
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
            "Choose output image path",
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
                "CUDA ready.\n"
                f"{status_text}"
            )
        else:
            self.environment_label.setStyleSheet(
                "background: #fbe8e8; color: #8a1f1f;"
            )
            self.environment_label.setText(
                "Execution blocked until CUDA is available.\n"
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
            preview_pane.clear_preview("No file selected.")
            return

        path = Path(raw_path).expanduser()
        if not path.exists() or not path.is_file():
            preview_pane.clear_preview(f"File not found: {path}")
            return

        try:
            preview_bytes = load_preview_png_bytes(path)
        except Exception as exc:  # pragma: no cover - preview is best-effort UI behavior
            preview_pane.clear_preview(f"Preview unavailable: {exc}")
            return

        preview_pane.set_preview_bytes(preview_bytes, str(path))

    def _collect_run_request(self) -> StyleTransferRunRequest:
        """Read the current UI values and validate them into a run request."""
        content_path = validate_image_path(
            self.content_input.text().strip(),
            "content image",
        )
        style_path = validate_image_path(
            self.style_input.text().strip(),
            "style image",
        )
        mask_path = validate_optional_image_path(
            self.mask_input.text().strip(),
            "mask image",
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
            self._show_error("Invalid configuration", str(exc))
            return

        self.result_preview.clear_preview("Waiting for the current run to finish.")
        self._set_running_state(True)
        self._set_status("Starting worker thread...", 0, "Preparing")

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
        self._set_status("Cancellation requested. Waiting for a safe checkpoint...", self.progress_bar.value())
        self.cancel_requested.emit()

    def _handle_worker_progress(self, progress: StyleTransferRunProgress) -> None:
        """Reflect worker progress updates in the GUI."""
        message = progress.message
        if progress.content_loss is not None and progress.style_loss is not None:
            message = (
                f"{progress.message}\n"
                f"Content loss: {progress.content_loss:.4f} | "
                f"Style loss: {progress.style_loss:.4f}"
            )
        self._set_status(message, progress.percent)

    def _handle_worker_success(self, result: StyleTransferRunResult) -> None:
        """Handle a completed background NST run."""
        self.result_preview.set_preview_bytes(
            result.preview_png_bytes,
            str(result.output_image_path),
        )
        self.output_summary.setPlainText(
            "\n".join(
                [
                    result.metadata_summary,
                    f"Content loss: {result.content_loss:.6f}",
                    f"Style loss: {result.style_loss:.6f}",
                    f"Mask applied: {result.applied_mask}",
                ]
            )
        )
        self._set_status("Run completed successfully.", 100, "Complete")
        self._set_running_state(False)

    def _handle_worker_failure(self, message: str) -> None:
        """Handle a background worker failure."""
        self.result_preview.clear_preview("Run failed before a new result was produced.")
        self.output_summary.setPlainText(message)
        self._show_error("Run failed", message)
        self._set_status("Run failed. Review the error message and adjust the inputs.", 0, "Failed")
        self._set_running_state(False)

    def _handle_worker_cancelled(self, message: str) -> None:
        """Handle a cooperatively cancelled NST run."""
        self.result_preview.clear_preview("The current run was cancelled.")
        self._set_status(message or "Run cancelled.", 0, "Cancelled")
        self.output_summary.setPlainText("The current run was cancelled before completion.")
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
