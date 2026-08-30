FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir huggingface_hub hf_transfer
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.1-schnell', allow_patterns=['transformer/*','text_encoder/*','text_encoder_2/*','vae/*','scheduler/*','tokenizer/*','tokenizer_2/*','*.json','*.txt'])"
RUN pip install --no-cache-dir runpod "diffusers>=0.32,<0.36" "transformers>=4.44" accelerate safetensors sentencepiece protobuf
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
