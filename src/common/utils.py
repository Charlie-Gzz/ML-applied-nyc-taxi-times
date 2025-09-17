from typing import Any, Dict
import os

def getenv(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)
