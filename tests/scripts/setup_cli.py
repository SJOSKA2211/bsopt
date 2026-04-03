#!/usr/bin/env python3
"""
Setup script for Black-Scholes CLI

Installs CLI as 'bsopt' command globally or in virtual environment.

Usage:
    python setup_cli.py install
    python setup_cli.py develop  # Development mode
    python setup_cli.py uninstall
"""

import subprocess
import sys
from pathlib import Path


def install_cli(dev_mode=False):
    """Install CLI package."""
    print("Installing Black-Scholes CLI...")

    # Ensure we're in the project directory
    project_dir = Path(__file__).parent

    # Install in editable mode for development
    if dev_mode:
        cmd = [sys.executable, "-m", "pip", "install", "-e", str(project_dir)]
        print("Installing in development mode...")
    else:
        cmd = [sys.executable, "-m", "pip", "install", str(project_dir)]
        print("Installing...")

    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Installation successful!")
        print("\nYou can now use 'bsopt' command:")
        print("  bsopt --help")
        print("  bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05")
        print("  bsopt auth login")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        sys.exit(1)


def uninstall_cli():
    """Uninstall CLI package."""
    print("Uninstalling Black-Scholes CLI...")

    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "bsopt"]

    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Uninstallation successful!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Uninstallation failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python setup_cli.py [install|develop|uninstall]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "install":
        install_cli(dev_mode=False)
    elif command == "develop" or command == "dev":
        install_cli(dev_mode=True)
    elif command == "uninstall":
        uninstall_cli()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python setup_cli.py [install|develop|uninstall]")
        sys.exit(1)


if __name__ == "__main__":
    main()
