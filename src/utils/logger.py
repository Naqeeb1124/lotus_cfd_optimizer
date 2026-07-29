"""JSON-line logger for the autonomous CFD design framework.

Each log entry is a single JSON object written to a line-delimited file.
Fields:
- run_id: UUID generated at the start of a CFD run.
- timestamp: ISO-8601 UTC timestamp.
- level: DEBUG/INFO/WARNING/ERROR.
- stage: CAD/MESH/SOLVER/POSTPROC/ARCHIVE.
- message: Human-readable description.
- details: Arbitrary key/value map for numeric results or file paths.
"""

import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any

# Generate a unique run ID
RUN_ID = str(uuid.uuid4())

# Log directory
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"run_{RUN_ID}.jsonl"

def _write(entry: Dict[str, Any]) -> None:
    """Write a JSON line to the log file."""
    entry.setdefault("run_id", RUN_ID)
    entry.setdefault("timestamp", datetime.datetime.utcnow().isoformat() + "Z")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def debug(stage: str, message: str, **details: Any) -> None:
    """Log a DEBUG-level message."""
    _write({"level": "DEBUG", "stage": stage, "message": message, "details": details})

def info(stage: str, message: str, **details: Any) -> None:
    """Log an INFO-level message."""
    _write({"level": "INFO", "stage": stage, "message": message, "details": details})

def warning(stage: str, message: str, **details: Any) -> None:
    """Log a WARNING-level message."""
    _write({"level": "WARNING", "stage": stage, "message": message, "details": details})

def error(stage: str, message: str, **details: Any) -> None:
    """Log an ERROR-level message."""
    _write({"level": "ERROR", "stage": stage, "message": message, "details": details})