from __future__ import annotations
 
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
 
from app.core.logger import get_logger
 
logger = get_logger(__name__)

class FeedbackStore:
    def __init__(self, path: str = "data/feedback.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def add(self, question: str, answer: str, rating: int, comment: str | None = None,) -> str:
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "rating": rating,
            "comment": comment,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        logger.info("Recorded feedback %s (rating=%s)", record["id"], rating)
        return record["id"]