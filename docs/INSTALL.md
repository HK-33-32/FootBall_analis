# Installation and first run

## Standalone application

This is the supported path for every clean GitHub clone. Install Docker with the
Compose v2 plugin, clone the repository, enter its directory and run:

```powershell
.\scripts\quickstart.ps1
```

or on Linux/macOS:

```bash
sh scripts/quickstart.sh
```

The script copies `.env.example` to the ignored `.env` only when `.env` does not
exist, creates ignored runtime directories, builds the pinned Python application,
starts only `api`, waits for its health endpoint and prints the browser URL. It
does not download models or send data to a remote service. Re-running it is safe.
The Docker runtime dependency resolution is recorded in `requirements-docker.lock`;
Python and direct application dependencies are pinned in the image definition and
`pyproject.toml`.

Manual equivalent:

```bash
cp .env.example .env                 # PowerShell: Copy-Item .env.example .env
docker compose up -d --build api
curl http://localhost:8080/api/v1/health
```

Expected standalone health includes `"status":"ok"`, perception
`"not_configured"`, and semantic VLM `"configured":false`. The UI and saved-report
viewer work; submitting a new analysis requires perception and fails explicitly
while it is absent.

## Full GPU analysis

The optimized perception runtime is included in `services/perception` and builds
directly from the clone. Large model files remain external and retain their own
terms. The default source build uses the trained CLIP jersey reader and does not
compile the optional local GGUF reader.

Requirements:

- NVIDIA driver and Docker GPU support;
- a local checkpoint directory (default `./weights`) containing the filenames
  downloaded/checked by the GPU quick start;
- about 16 GB VRAM for perception alone. Do not concurrently start the optional
  large VLM on a single 16 GB GPU.

Windows one-command build, weight download, verification and launch:

```powershell
.\scripts\quickstart-gpu.ps1
```

This is the preferred online installation. It downloads the three checkpoints
required by the default CLIP runtime; RF-DETR populates its mounted cache on first
analysis. The optional local jersey Qwen and semantic-event Qwen are not required.

### Full offline weight bundle

For another machine or an internal release, keep all model files in one external
uncompressed TAR. The verified full manifest contains nine files (about 10.5 GiB):
perception checkpoints, RF-DETR and the optional local Qwen GGUF/projector. Model
files are already compressed internally, so ZIP adds time with little size benefit.

Create the bundle on a machine whose `weights/` contains the verified models:

```powershell
python scripts/weights_bundle.py pack `
  --weights-dir .\weights `
  --output D:\releases\football-intelligence-weights-2026-09-09.tar
```

The command also writes a `.sha256` sidecar. Transfer both files separately from
Git. After cloning on the target Windows machine, install, verify and start with:

```powershell
.\scripts\install-gpu.ps1 `
  D:\releases\football-intelligence-weights-2026-09-09.tar `
  -Sha256 <digest-from-the-sidecar>
```

Linux uses the same TAR and manifest:

```bash
sh scripts/install-gpu.sh /mnt/models/football-intelligence-weights-2026-09-09.tar \
  <digest-from-the-sidecar>
```

The installer rejects extra/archive-traversal entries, verifies the embedded
manifest, exact sizes and SHA-256 of every model, and refuses to overwrite a
different existing checkpoint unless `--replace` (Python CLI) or `-Replace`
(PowerShell) is explicit. Use `-NoStart` to install without launching containers.
Python 3 is needed only for this offline bundle flow; the online Docker quick start
does not require host Python.

Do not upload the archive to Git. Publish it as a private/internal release artifact
or an object-storage download only after verifying redistribution rights for every
listed model. The repository's Apache license does not grant those rights.

Manual equivalent:

```powershell
docker compose --profile full build perception
docker compose --profile full run --rm perception fetch-weights
.\scripts\check_gpu_prerequisites.ps1
$env:FI_CORE_URL="http://perception:8000"
$env:FI_PERCEPTION_URL="http://perception:8000"
docker compose --profile full up -d --build api perception
```

Weights and raw datasets are mounted and are never copied into Git or the image.
Check:

```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/v1/health
Invoke-RestMethod http://localhost:8080/api/v1/health
```

`FI_PERCEPTION_IMAGE` changes the output tag. Set `FI_WEIGHTS_DIR` in `.env` for
a checkpoint/cache directory outside the repository. Set
`FI_PERCEPTION_STAGE=runtime-vlm` only when the optional CUDA llama.cpp/GGUF jersey
reader is required; this adds a substantial compile step.

## Optional semantic VLM

The `vlm` service is optional and intentionally not started by quick start. It can
download a large pinned Qwen model and requires a separate memory budget. On a
suitable machine set `FI_VLM_BASE_URL=http://vlm:8000` and explicitly include the
service:

```bash
docker compose --profile full up -d vlm
```

The default `HF_HOME=./weights/huggingface` keeps this cache beside the other
external models, but it is not part of the nine-file offline bundle because vLLM
can fetch it from the pinned Hugging Face revision. Do not put access tokens in Compose or Git. Store local overrides only in `.env`,
which is ignored. The default local server key `EMPTY` is not an external secret.

## Native Python development

Python 3.11 or 3.12 and FFmpeg are required:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: . .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
football-intelligence serve --host 127.0.0.1 --port 8080
```

## Data and licenses

Raw match footage, runtime outputs, weights and dataset archives are ignored.
SoccerNet access must use its official download flow and terms; see
[`data/README.md`](../data/README.md). Do not commit generated reports containing
licensed footage. Repository code is Apache-2.0 except separately licensed or
excluded dependencies described in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
