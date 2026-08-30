FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir runpod "diffusers>=0.32,<0.36" "transformers>=4.44" accelerate safetensors sentencepiece protobuf huggingface_hub hf_transfer
RUN pip install --no-cache-dir "peft>=0.11,<0.16"
RUN python -c "import peft; from diffusers.utils import USE_PEFT_BACKEND; print('peft', peft.__version__, 'backend', USE_PEFT_BACKEND); assert USE_PEFT_BACKEND, 'PEFT backend NOT enabled'"
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]


