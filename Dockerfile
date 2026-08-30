FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hfcache
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# пин ai-toolkit на torch-2.6-совместимый коммit (апр 2025) — избегаем torch-2.13-only op
RUN git clone https://github.com/ostris/ai-toolkit /ai-toolkit && \
    cd /ai-toolkit && git checkout 5890e67a46 && git submodule update --init --recursive
# requirements пинят torch==2.6.0 (уже в базе, cu124) — torch не переустанавливаем
RUN cd /ai-toolkit && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod pyyaml
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
