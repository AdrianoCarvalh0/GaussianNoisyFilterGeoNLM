# GaussianExperiments

Reproducible Gaussian denoising experiments with **RAPIDS + CuPy** (GPU) and **BM3D** (CPU).  
The dev environment is **frozen** via a Conda **explicit spec** (`conda-spec-linux-64.txt`) plus a minimal `requirements-pip.txt`—no environment solving during setup.

---

## Requirements

- **Docker**
- **NVIDIA GPU** + driver compatible with CUDA **12.2** (optional but recommended)
- **VS Code** + **Dev Containers** extension (or use plain Docker CLI)

> **Windows tip:** use **WSL (Ubuntu)** and open the repository from the WSL path (avoid `\\wsl.localhost\...`).

---

## Quick Start (VS Code)

1. Open the folder in VS Code.  
2. Press **Ctrl+Shift+P** → _Dev Containers: Rebuild and Reopen in Container_.  
   - The image is built from the `Dockerfile`.  
   - The Conda environment **`gaussian`** is created from `conda-spec-linux-64.txt`.  
   - Pip-only extras (e.g., `bm3d`) are installed from `requirements-pip.txt`.

3. Sanity check:
   ```bash
   python - << 'PY'
   import cupy as cp, numpy as np, skimage, pandas as pd, sklearn, networkx as nx, bm3d
   print("GPUs:", cp.cuda.runtime.getDeviceCount(), "| Environment OK")
   PY
   ```

---

## Quick Start (Docker CLI)

```bash
# Build the image
docker build -t gaussian-frozen .

# Open an interactive shell in the container
docker run --gpus=all --shm-size=4g -it --rm   -v "$PWD":/workspace -w /workspace gaussian-frozen bash
```

---

## Repository Layout

```
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
```

This `data/output` layout lets users run scripts and immediately see images written to `test/{bm3d,nlm,geonlm}` per noise profile (`low_noisy`, `moderate_noisy`, `high_noisy`).

---

## Running Experiments

Notebooks Test src/gaussian_experiments ==> Gaussian_low.ipynb

<!-- If you use a driver like `src/run_benchmarks.py`, run it directly:

```bash
python -m src.run_benchmarks   --sigma-noise 0.05 --sigma-gauss 1.0 --seed 7   --out data/output/moderate_noisy/test/nlm/out.json
``` -->

You can adapt arguments and output paths for **BM3D**, **NLM**, and **GEO-NLM**.  
(Optionally add a `Makefile` with targets like `make dirs`, `make demo-low`, etc.)

---

## Reproducibility & Environment

- The environment is frozen by:
  - `conda-spec-linux-64.txt` (generated with `conda list --explicit --md5`)
  - `requirements-pip.txt` (minimal pip-only packages; avoid duplicating Conda-managed libs)
- After changing packages **inside the container**, refresh the lockfile:
  ```bash
  conda list --explicit --md5 > conda-spec-linux-64.txt
  # If you added/changed pip-only packages, keep requirements-pip.txt minimal.
  ```
- For paper-ready runs, pin key versions (e.g., `cupy==13.*`, `scikit-image==0.24.*`).

---

## Data & Outputs

Large artifacts (images, `.npy`, `.npz`) can inflate the repo size. Consider **Git LFS**:

```bash
git lfs install
echo "data/** filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
```

---

## Troubleshooting

- **Workspace write errors:** the container runs as **root** and sets permissive ACLs to avoid WSL/NTFS issues.
- **No GPU detected:** check `nvidia-smi` on the host and ensure Docker GPU support is enabled (Docker Desktop → Resources → WSL Integration).
- **Dev Containers solving env:** avoided here by using a Conda explicit spec. If you need flexibility later, modify the env inside the container and re-export the spec.

---

## License

License: [MIT](./LICENSE)  
SPDX-Identifier: `MIT`
