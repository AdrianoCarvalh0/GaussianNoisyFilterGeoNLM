GaussianExperiments

Reproducible experiments for Gaussian denoising with RAPIDS + CuPy (GPU) and BM3D (CPU).
The development environment is frozen using a Conda explicit spec (conda-spec-linux-64.txt) plus a minimal requirements-pip.txt—so there’s no environment “solving” when you open the project.

Requirements

Docker (recommended)

NVIDIA GPU + driver compatible with CUDA 12.2 (optional but recommended)

VS Code + Dev Containers extension (or use plain Docker CLI)

Tip (Windows): work inside WSL (Ubuntu) and open the repo from there (avoid \\wsl.localhost\... paths).

Getting Started (VS Code)

Open the folder in VS Code.

Command Palette → Dev Containers: Rebuild and Reopen in Container.

The container image is built from the Dockerfile.

The Conda environment gaussian is created from conda-spec-linux-64.txt.

Pip-only extras (e.g., bm3d) come from requirements-pip.txt.

Sanity check:

python - << 'PY'
import cupy as cp, numpy as np, skimage, pandas as pd, sklearn, networkx as nx, bm3d
print("GPUs:", cp.cuda.runtime.getDeviceCount(), "| Environment OK")
PY

Getting Started (Docker CLI only)
# Build image
docker build -t gaussian-frozen .

# Run an interactive shell
docker run --gpus=all --shm-size=4g -it --rm \
  -v "$PWD":/workspace -w /workspace gaussian-frozen bash

Repository Layout
GaussianExperiments/
├─ .devcontainer/            # devcontainer.json
├─ data/
│  ├─ input/                 # (optional) raw images (usually gitignored)
│  └─ output/                # generated results & figures
│     ├─ low_noisy/
│     │  └─ test/{bm3d,nlm,geonlm}/
│     ├─ moderate_noisy/
│     │  └─ test/{bm3d,nlm,geonlm}/
│     └─ high_noisy/
│        └─ test/{bm3d,nlm,geonlm}/
├─ src/                      # source code (filters, metrics, runners)
├─ Dockerfile                # frozen environment image
├─ conda-spec-linux-64.txt   # Conda explicit spec (lockfile)
├─ requirements-pip.txt      # minimal pip-only deps (e.g., bm3d)
├─ .dockerignore
└─ README.md


The data/output tree above is organized so users can run scripts and immediately see images written into test/{bm3d,nlm,geonlm} for each noise profile (low_noisy, moderate_noisy, high_noisy).

Running Experiments

If you have a driver/script like src/run_benchmarks.py, you can run it directly:

python -m src.run_benchmarks --sigma-noise 0.05 --sigma-gauss 1.0 --seed 7 \
  --out data/output/moderate_noisy/test/nlm/out.json


Adjust arguments and output paths as you like (for BM3D, NLM, GEO-NLM, etc.).
If you use a Makefile, you can add handy targets like make dirs, make demo-low, etc.

Reproducibility & Environment

The environment is frozen via:

conda-spec-linux-64.txt (generated with conda list --explicit --md5)

requirements-pip.txt (minimal pip-only packages; avoid listing Conda-managed libs here)

To update the lockfile after changing packages inside the container:

conda list --explicit --md5 > conda-spec-linux-64.txt
# If you added/changed pip-only packages, keep requirements-pip.txt minimal


Best practice: keep versions pinned (e.g., cupy==13.*, scikit-image==0.24.*) when finalizing figures for a paper.

Data & Outputs

Large binary artifacts (images, .npy, .npz) can bloat the repo.
Consider Git LFS:

git lfs install
echo "data/** filter=lfs diff=lfs merge=lfs -text" >> .gitattributes


If you publish with data included, ensure data/input/ and data/output/ have a clear subfolder convention (as shown above) so users can reproduce plots easily.

Troubleshooting

Container cannot write to workspace: we run the container as root and set permissive workspace ACLs to avoid WSL/NTFS permission issues.

No GPU detected: check nvidia-smi on the host and Docker → GPU support. In WSL, enable GPU in Docker Desktop and WSL integration for your distro.

Dev Containers stuck “solving environment”: this setup avoids solving by using a Conda explicit spec. If you really need flexibility later, regenerate the spec after testing changes.

License: [MIT](./LICENSE)