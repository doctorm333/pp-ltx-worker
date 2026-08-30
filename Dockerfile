FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hfcache
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ostris/ai-toolkit /ai-toolkit && \
    cd /ai-toolkit && git submodule update --init --recursive
# патчим list[torch.Tensor] -> typing.List[torch.Tensor] (torch 2.6 infer_schema не принимает builtin list)
COPY patch_annotations.py /tmp/patch_annotations.py
RUN python /tmp/patch_annotations.py
# GUARD: убеждаемся что ломающих аннотаций больше нет
RUN ! grep -rn "list\[torch.Tensor\]" /ai-toolkit/toolkit/ || (echo "STILL HAS list[torch.Tensor]" && exit 1)
RUN cd /ai-toolkit && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod pyyaml
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
