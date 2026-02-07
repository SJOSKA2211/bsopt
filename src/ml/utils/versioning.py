import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def calculate_data_hash(filepath: str) -> str:
    """Calculate SHA256 hash of a data file for versioning."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def tag_dataset(data_dir: str, version_name: str | None = None):
    """
    Create a version tag for a dataset.
    Simulates DVC functionality.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory {data_dir} does not exist.")
        return

    version = version_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata: dict[str, Any] = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "files": [],
    }

    for file in data_path.glob("*.csv"):
        metadata["files"].append(
            {
                "name": file.name,
                "hash": calculate_data_hash(str(file)),
                "size": file.stat().st_size,
            }
        )

    version_file = data_path / f"version_{version}.json"
    with open(version_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Tagged dataset version {version} in {version_file}")
    return version
