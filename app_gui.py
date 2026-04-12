"""Desktop entry point for the PySide6 application."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from neural_style.config import APP_TITLE, APP_WINDOW_HEIGHT, APP_WINDOW_WIDTH
from neural_style.validation import build_startup_status_message, is_cuda_ready


class MainWindow(QMainWindow):
    """Minimal main window scaffold for later GUI expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(APP_WINDOW_WIDTH, APP_WINDOW_HEIGHT)
        status_text = build_startup_status_message()
        if is_cuda_ready():
            label_text = (
                "GUI scaffold ready. Full controls will be added later.\n\n"
                f"{status_text}"
            )
        else:
            label_text = (
                "GUI scaffold ready, but execution is blocked until a CUDA-capable "
                "environment is available.\n\n"
                f"{status_text}"
            )
        self.setCentralWidget(QLabel(label_text))


def main() -> int:
    """Start the Qt event loop."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
