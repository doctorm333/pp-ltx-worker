FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hfcache
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# clone + checkout старого коммита (torch 2.6, без nvfp4) — отдельными слоями чтобы точно применилось
RUN git clone https://github.com/ostris/ai-toolkit /ai-toolkit
RUN cd /ai-toolkit && git checkout 5890e67a46b8dd05f97b0c8f7cf9a5a1c4f6f4e0 || git checkout 5890e67a46
RUN cd /ai-toolkit && git submodule update --init --recursive
# GUARD: гарантия что мы на старом коммите (nvfp4-файла быть не должно) — иначе сборка падает
RUN test ! -f /ai-toolkit/toolkit/util/convrot_quant.py
RUN cd /ai-toolkit && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod pyyaml
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
