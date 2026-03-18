#!/usr/bin/env python3
"""
Initialize SHM Buffers.
Ensures all required shared memory segments are pre-allocated and zeroed.
"""

import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.shm_init import SHM_CONFIGS, initialize_all_shm


def main():
    parser = argparse.ArgumentParser(description="Initialize BS-OPT SHM Buffers")
    parser.add_argument("--force", action="store_true", help="Unlink existing SHM before creating")
    parser.add_argument("--list", action="store_true", help="List configured SHM segments")

    args = parser.parse_args()

    if args.list:
        print(f"{'Name':<30} | {'Size (MB)':<10} | {'Description'}")
        print("-" * 70)
        for config in SHM_CONFIGS:
            print(
                f"{config['name']:<30} | {config['size'] / (1024 * 1024):<10.2f} | {config['description']}"
            )
        return

    print("Initializing SHM Buffers...")
    initialize_all_shm(force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
