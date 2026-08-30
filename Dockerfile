FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
# --- stable layer: downloader deps + model weights (cached across app/handler tweaks) ---
RUN pip install --no-cache-dir huggingface_hub hf_transfer
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Lightricks/LTX-Video', allow_patterns=['transformer/*','text_encoder/*','vae/*','scheduler/*','tokenizer/*','*.json','*.txt'])"
# --- app deps: pinned to versions compatible with torch 2.4 ---
# diffusers 0.33+ registers a torch.library attention op (q/k/v) that torch 2.4's
# infer_schema rejects -> import crash. 0.32.2 has stable LTX without that op.
RUN pip install --no-cache-dir runpod "diffusers==0.32.2" "transformers==4.46.3" "accelerate>=0.34" safetensors sentencepiece protobuf imageio imageio-ffmpeg
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
