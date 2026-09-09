# Source-built perception service. Weights and match data are mounted, never baked in.
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
ARG PERCEPTION_STAGE=runtime

FROM ${CUDA_DEVEL_IMAGE} AS vlm-builder
ARG CUDA_ARCHS=89;120
ARG LLAMA_CPP_VERSION=0.3.34
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN mkdir -p /wheels \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv build-essential cmake ninja-build \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/build-venv \
    && /opt/build-venv/bin/pip install \
        pip==26.2.1 setuptools==78.1.0 wheel==0.48.0 scikit-build-core==0.11.6 \
    && CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
       /opt/build-venv/bin/pip wheel --no-deps -w /wheels \
       llama-cpp-python==${LLAMA_CPP_VERSION}

FROM ${RUNTIME_IMAGE} AS runtime
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/weights/torch \
    HF_HOME=/opt/weights/huggingface \
    RF_HOME=/opt/weights/rfdetr \
    FG_CHECKPOINT_DIR=/opt/weights/checkpoints

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv ffmpeg ca-certificates git \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv

RUN pip install pip==26.2.1 setuptools==78.1.0 wheel==0.48.0 \
    && pip install --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    && python -c "import torch; assert torch.version.cuda, 'installed CPU-only torch'"

RUN find /opt/venv/lib/python3.12/site-packages/nvidia -type d -name lib \
        > /etc/ld.so.conf.d/torch-cuda.conf \
    && echo /opt/venv/lib/python3.12/site-packages/torch/lib \
        >> /etc/ld.so.conf.d/torch-cuda.conf \
    && ldconfig

COPY services/perception/requirements.txt /tmp/requirements.txt
COPY services/perception/constraints.lock /tmp/constraints.lock
RUN pip install --requirement /tmp/requirements.txt --constraint /tmp/constraints.lock \
    && pip check

# Keep the measured full-integrity RF-DETR hash implementation.
COPY docker/weight-io-overlay /opt/weight-io-overlay
COPY docker/apply_core_overlay.py /opt/apply_core_overlay.py
RUN python /opt/apply_core_overlay.py \
    --root /opt/venv/lib/python3.12/site-packages/rfdetr \
    --overlay /opt/weight-io-overlay

WORKDIR /app
COPY services/perception/engine /app/engine
COPY services/perception/core /app/core
COPY services/perception/rosters /app/rosters
COPY services/perception/scripts /app/scripts
RUN sed -i 's/\r$//' /app/scripts/*.sh \
    && chmod +x /app/scripts/*.sh \
    && python -m compileall -q /app/core /app/engine

ENV PYTHONPATH=/app:/app/engine:/app/engine/reid \
    FG_ENGINE_DIR=/app/engine \
    FG_STORAGE_DIR=/data \
    FG_ROSTER_DIR=/app/rosters \
    FG_HOST=0.0.0.0 \
    FG_PORT=8000
VOLUME ["/data", "/opt/weights"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/scripts/healthcheck.py || exit 1
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["serve"]

FROM runtime AS runtime-vlm
COPY --from=vlm-builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels && pip check

# Select runtime-vlm explicitly to compile the optional local GGUF jersey reader.
FROM ${PERCEPTION_STAGE} AS final
