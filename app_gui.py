"""Desktop entry point for the PySide6 application."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from neural_style.config import APP_TITLE


class MainWindow(QMainWindow):
    """Minimal main window scaffold for later GUI expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setCentralWidget(
            QLabel("GUI scaffold ready. Full controls will be added later.")
        )


def main() -> int:
    """Start the Qt event loop."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
