FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hfcache
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir runpod "git+https://github.com/huggingface/diffusers" transformers accelerate safetensors sentencepiece protobuf huggingface_hub hf_transfer
RUN python -c "from diffusers import ZImagePipeline; print('ZImagePipeline import OK')"
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
