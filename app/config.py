from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIR          = BASE_DIR / "data"
THUMBNAILS_DIR    = BASE_DIR / "static" / "thumbnails"
OUTPUT_DIR        = BASE_DIR / "output"
LABELED_FACES_DIR = BASE_DIR / "labeled-faces"
DB_PATH           = DATA_DIR / "library.db"
CHROMA_PATH       = DATA_DIR / "chroma"

KEYFRAME_INTERVAL_SEC = 1.0
THUMBNAIL_SIZE        = (128, 128)
FACE_DET_SIZE         = (640, 640)

MODEL_NAME_DEFAULT = "buffalo_sc"
MODEL_NAME_HIGH    = "buffalo_l"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}

COLLAGE_CELL_SIZE = 400   # px — each photo cell is resized to this square
COLLAGE_PADDING   = 8     # px — gutter between cells and canvas border
