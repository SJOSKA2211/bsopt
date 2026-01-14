# Anti-Freeze Guide: Solving System Saturation during Builds

This guide outlines strategies to prevent system "freezing" (CPU/RAM saturation) when building the heavy ML/AI stack (PyTorch, Ray, Qiskit) locally.

## 1. Throttle Build Concurrency (Immediate Fix)

By default, Docker Compose builds services in parallel, which can exhaust system resources. Force sequential builds to reduce load.

### Usage
```bash
export COMPOSE_PARALLEL_LIMIT=1
docker-compose build
```

You can add this to your shell profile (`.bashrc` or `.zshrc`):
```bash
export COMPOSE_PARALLEL_LIMIT=1
```

## 2. Limit Docker System Resources

Prevent Docker from consuming all available RAM.

### Mac/Windows
1. Go to **Docker Desktop Settings** -> **Resources**.
2. Set **CPU** to `Total - 2`.
3. Set **RAM** to `Total - 4GB` (leave OS breathing room).

### Linux
Create or edit `/etc/docker/daemon.json` to set default ulimits (note: direct resource limits are better handled via systemd slice properties or cgroups, but this helps prevent file descriptor exhaustion):

```json
{
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "cgroup-parent": "docker.slice"
}
```

## 3. Remote Builder (Expert Solution)

Offload the heavy compilation to a remote server or cloud VM while keeping your local context.

### Setup
1. **Create Context:**
   ```bash
   docker context create remote-builder --docker "host=ssh://user@remote-server-ip"
   ```

2. **Use Context:**
   ```bash
   docker context use remote-builder
   ```

3. **Build:**
   ```bash
   docker-compose build
   ```
   *The build runs on the remote server, but images are available to your local client context if configured, or you can push/pull from a registry.*

## 4. Cloud-Native Workflow (Recommended)

We are transitioning to a "Build in Cloud, Pull to Local" workflow.
- **Push** your code to GitHub.
- **Wait** for GitHub Actions to build the images.
- **Pull** the pre-built images locally:
  ```bash
  docker-compose pull
  docker-compose up
  ```
