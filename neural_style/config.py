"""Project-wide configuration defaults."""

from pathlib import Path

APP_TITLE = "Neural Style Transfer App"
APP_ORGANIZATION = "Course Project"
APP_WINDOW_WIDTH = 1280
APP_WINDOW_HEIGHT = 800

DEFAULT_IMAGE_SIZE = 768
DEFAULT_NUM_STEPS = 300
DEFAULT_STYLE_STRENGTH = 1.0
MIN_NUM_STEPS = 50
MAX_NUM_STEPS = 1000
MIN_STYLE_STRENGTH = 0.1
MAX_STYLE_STRENGTH = 10.0

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT_STEM = "result"
DEFAULT_METADATA_SUFFIX = ".json"

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
