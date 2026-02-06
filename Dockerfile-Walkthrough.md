# Dockerfile Walkthrough - TripoSG-WebUI

## Line 7: `FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

The "base image" - the starting point for our container. Like choosing an OS to install on a new computer.

Why this image: We need PyTorch 2.6.0 specifically (torch 2.10 breaks diso/DiffDMC). This image comes pre-loaded with:
- Python
- PyTorch 2.6.0
- CUDA 12.4 runtime (GPU drivers talk through this)
- cuDNN 9 (neural network acceleration library)

We use `-runtime` not `-devel` because we don't need to compile CUDA code, just run it. Runtime is smaller (~6GB vs ~10GB).

---

## Line 9: `WORKDIR /app`

Sets the working directory inside the container. All subsequent commands run from `/app`. Same reason you `cd` into a project directory.

---

## Lines 12-19: System dependencies

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
```

Installs OS-level libraries that Python packages need but can't install themselves.

Why each one:
- `git` - needed on line 22 to clone the TripoSG repo
- `libgl1-mesa-glx` - OpenCV needs this (OpenGL rendering)
- `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1` - more OpenCV/GUI dependencies

`--no-install-recommends` skips optional packages to keep image smaller. `rm -rf /var/lib/apt/lists/*` cleans up the package cache. Both are Docker best practices to minimize image size.

The `&&` chaining: Docker creates a "layer" for each `RUN` instruction. By chaining commands with `&&`, we keep it all in one layer. If we did `RUN apt-get update` and then `RUN apt-get install...` separately, the cache from update would be in a permanent layer even after cleanup.

---

## Lines 22-25: Clone TripoSG source

```dockerfile
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSG.git /app/triposg-src \
    && mv /app/triposg-src/scripts /app/scripts \
    && mv /app/triposg-src/triposg /app/triposg \
    && rm -rf /app/triposg-src
```

Grabs the original TripoSG model code, keeps only what we need (`scripts/` and `triposg/`), deletes the rest.

Our WebUI (`app.py`) imports from the TripoSG model code but we don't ship it in our repo (it's their code, not ours). So we fetch it at build time.

`--depth 1`: Only downloads the latest commit, not the full git history. Saves time and space.

Potential issue: If VAST-AI changes their repo structure, this breaks. A more robust approach would pin to a specific commit hash.

---

## Lines 29-31: Python dependencies

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir flask \
    && pip install --no-cache-dir -r requirements.txt
```

Copies our requirements file in, then installs Flask + everything in requirements.txt.

Why COPY before RUN: Docker caches layers. If `requirements.txt` hasn't changed, Docker skips this step on rebuild. If we copied all files first, any change to `app.py` would invalidate the cache and reinstall all packages. This ordering is a deliberate optimization.

`--no-cache-dir`: Tells pip not to cache downloaded packages. Inside a container, you'll never `pip install` again, so the cache is just wasted space.

---

## Lines 34-36: Copy application files

```dockerfile
COPY app.py .
COPY 3d-printing-logo.png eak.png ./
COPY TripoSG-WebUI_Parameters-explained.txt ./
```

Copies our actual app and static assets into the container.

Why last: These files change most frequently. By copying them last, we maximize Docker's layer cache. Changing `app.py` only rebuilds from this point forward, not from the beginning.

---

## Lines 39-44: Outputs and volume

```dockerfile
RUN mkdir -p /app/outputs
VOLUME /app/pretrained_weights
```

Creates the outputs directory and declares `/app/pretrained_weights` as a volume mount point.

Why VOLUME: The model weights are ~30GB. Without a volume, they'd download every time you create a new container. The `VOLUME` instruction tells Docker "this directory should be stored outside the container's filesystem." Data here persists across container restarts and rebuilds.

---

## Lines 46-48: Expose and run

```dockerfile
EXPOSE 5000
CMD ["python", "app.py"]
```

`EXPOSE 5000`: Documentation that the app listens on port 5000. It doesn't actually open the port - that's done at runtime with `-p 5000:5000`.

`CMD`: The default command when the container starts. The bracket syntax (`["python", "app.py"]`) is called "exec form" - it runs the process directly without a shell wrapper, which means signals (like Ctrl+C) go straight to Python.

---

# docker-compose.yml Walkthrough

Compose wraps all those `docker run` flags into a declarative YAML file. Instead of remembering a long command, you just `docker compose up`.

## Lines 11-13: Service definition

```yaml
build: .
image: triposg-webui:latest
container_name: triposg-webui
```

`build: .` - Build from the Dockerfile in the current directory.
`image` - Tag the built image with this name.
`container_name` - Fixed name so you can `docker logs triposg-webui` instead of a random hash.

## Line 14: `restart: unless-stopped`

If the container crashes, Docker automatically restarts it. Also starts it on system boot. Only stops if you explicitly `docker compose down`.

Why: Production-style behavior. If the Flask app crashes from an OOM or bug, it comes right back.

## Lines 16-22: Ports and volumes

```yaml
ports:
  - "5000:5000"
volumes:
  - triposg-weights:/app/pretrained_weights
  - triposg-outputs:/app/outputs
```

`"5000:5000"` - Maps host port 5000 to container port 5000. Format is `host:container`. You could do `"8080:5000"` to access it on port 8080 from your browser.

Named volumes: `triposg-weights` and `triposg-outputs` are Docker-managed volumes. Docker stores them in `/var/lib/docker/volumes/`. They persist even if you delete the container.

## Lines 23-29: GPU passthrough

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

The compose v2 way to say `--gpus all`. Tells Docker to pass through 1 NVIDIA GPU to the container via nvidia-container-toolkit.

`count: 1` - Only pass one GPU. We could say `all` for both Titan Vs, but TripoSG only uses one.

## Lines 31-36: Health check

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

Docker periodically pings the Flask app to check if it's alive.

Potential issue: The container doesn't have `curl` installed. The base pytorch image may not include it. We may need to install curl in the Dockerfile or switch to a Python-based health check.

`start_period: 60s` - Gives the app 60 seconds to start up before health checks count as failures. Important because model loading can take a while.

---

# Known Issues

1. **Health check uses `curl`** but we never install curl in the Dockerfile. Will likely fail.
2. **Git clone not pinned** to a commit - if upstream repo changes, build could break.
