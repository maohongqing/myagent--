from __future__ import annotations

import warnings
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(BACKEND_DIR)

LOG_DIR = BACKEND_DIR / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.stdout = (LOG_DIR / "uvicorn-runner.out.log").open("a", encoding="utf-8", buffering=1)
sys.stderr = (LOG_DIR / "uvicorn-runner.err.log").open("a", encoding="utf-8", buffering=1)
warnings.filterwarnings("ignore", category=Warning)

import uvicorn  # noqa: E402

uvicorn.run(
    "myagent.backend.app.main:app",
    host="127.0.0.1",
    port=8000,
    log_level="info",
)
