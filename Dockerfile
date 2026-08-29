FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir runpod "diffusers>=0.32.0" transformers accelerate safetensors imageio imageio-ffmpeg sentencepiece hf_transfer
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Lightricks/LTX-Video')"
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
