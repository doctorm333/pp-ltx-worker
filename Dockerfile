FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*
# stable layer: downloader + Wan2.2-5B weights (кэшируется между правками)
RUN pip install --no-cache-dir huggingface_hub hf_transfer
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.2-TI2V-5B-Diffusers', allow_patterns=['transformer/*','text_encoder/*','vae/*','scheduler/*','tokenizer/*','*.json','*.txt'])"
# app deps: diffusers из git main (Wan2.2 5B требует), torch 2.6 переваривает attention custom-op
RUN pip install --no-cache-dir runpod "git+https://github.com/huggingface/diffusers" "transformers>=4.49" accelerate safetensors sentencepiece protobuf ftfy imageio imageio-ffmpeg
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
