FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY project/ ./project/
COPY model_v2_result/inceptionresnet_v2/ ./model_v2_result/inceptionresnet_v2/

EXPOSE 8002

CMD ["uvicorn", "project.main:app", "--host", "0.0.0.0", "--port", "8002"]
