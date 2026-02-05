import os
from pathlib import Path


def sanitize_path(base_dir: Path, user_path: str) -> Path:
    """
    Sanitizes a user-provided path to ensure it stays within a designated base directory.

    Args:
        base_dir: The root directory that user_path must be confined within.
        user_path: The path provided by the user.

    Returns:
        A Path object representing the sanitized path within base_dir.

    Raises:
        ValueError: If the user_path attempts to access outside of the base_dir.
    """
    if not base_dir.is_absolute():
        base_dir = base_dir.resolve()

    # Resolve the user's intended path
    # Use os.path.join and Path() for robust path handling across OS
    full_path = Path(os.path.join(base_dir, user_path)).resolve()

    # Ensure the resolved path is a sub-path of the base directory
    if not full_path.is_relative_to(base_dir):
        raise ValueError(f"Path traversal detected: {user_path} attempts to access outside {base_dir}")

    return full_path

