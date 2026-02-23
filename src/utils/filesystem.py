import os
from pathlib import Path


def sanitize_path(base_dir: Path, user_path: str) -> Path:
    """
    Sanitized path protection.
    OPTIMIZED: Reduced syscall overhead for simple paths.
    """
    # Quick check for obvious traversal attempts
    if ".." in user_path or user_path.startswith("/"):
        # Fallback to full resolve for suspicious paths
        full_path = Path(os.path.join(base_dir, user_path)).resolve()
    else:
        # Fast path for simple relative strings
        full_path = base_dir / user_path

    if not full_path.is_relative_to(base_dir):
        raise ValueError(f"Path traversal detected: {user_path}")

    return full_path

async def sanitize_path_async(base_dir: Path, user_path: str) -> Path:
    """Non-blocking path sanitization."""
    from anyio.to_thread import run_sync
    return await run_sync(sanitize_path, base_dir, user_path)
