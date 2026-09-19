"""Project-relative configuration; secrets are never written by the application."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def data_root():
    return Path(os.environ.get("FEDRAG_DATA_DIR", ROOT)).expanduser().resolve()
