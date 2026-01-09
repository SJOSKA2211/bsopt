# Self-Hosted GPU Runner Setup for BS-Opt

This document describes how to set up a self-hosted GitHub Actions runner with GPU support for the Continuous Training (CT) job.

## Prerequisites
- Ubuntu 22.04+
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker installed

## Setup Steps

1. **Install NVIDIA Drivers:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-535
   ```

2. **Install NVIDIA Container Toolkit:**
   Follow the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

3. **Register GitHub Runner:**
   - Go to your repository on GitHub.
   - Settings > Actions > Runners > New self-hosted runner.
   - Follow the instructions for Linux.
   - **Label the runner:** `self-hosted`, `gpu`.

4. **Configure Runner for Docker:**
   Ensure the runner user has permissions to run Docker:
   ```bash
   sudo usermod -aG docker $USER
   ```

5. **Verify GPU in Docker:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

## CI/CD Workflow Integration
The `mlops-pipeline.yml` is configured to look for a runner with labels `[self-hosted, gpu]`. If found, it will execute the `pytorch` training with `--use-gpu`. If not, it falls back to a standard CPU-based run.
