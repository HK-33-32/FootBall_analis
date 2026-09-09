# Perception service

This is the source of the project's former standalone Football Core, now part of
the monorepo. It exposes the GPU perception API used by the application service:
field calibration, RF-DETR detection, tracking/ReID, role and jersey inference,
tracklet refinement, projection, rendering and low-level analytics.

The checked-in source was exported from the accepted
`football-core:clip-release-20260909` runtime. It already includes the measured
batching, calibration, projection-cache and direct-CLIP-checkpoint changes. The
only build-time patch left is the RF-DETR package's full-integrity weight hashing,
because that file belongs to the pinned third-party wheel.

Weights remain outside Git under `./weights`:

```text
weights/
  checkpoints/
    CLIP_Jersey.pth
    SoccernetGSR_EfficientNet_Best.pth
    sports_model.pth.tar-60
    Qwen2.5-VL-7B-Instruct-Q8_0.gguf          # optional
    mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf    # optional
  rfdetr/                                     # populated on first use
  torch/                                      # model cache
```

From the repository root:

```powershell
docker compose --profile full build perception
docker compose --profile full run --rm perception fetch-weights
.\scripts\check_gpu_prerequisites.ps1
$env:FI_CORE_URL="http://perception:8000"
$env:FI_PERCEPTION_URL="http://perception:8000"
docker compose --profile full up -d api perception
```

The default `runtime` image uses the trained CLIP head. Set
`FI_PERCEPTION_STAGE=runtime-vlm` to compile `llama-cpp-python` with CUDA for the
optional local GGUF jersey reader. That build is substantially slower and uses
`FI_CUDA_ARCHS` (`89;120` by default). The separate application `vlm` service is
for semantic event reasoning and is not the same model process.
