FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
       libgl1 \
       libglib2.0-0 \
       git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Normal pip install for req, no cache
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Build Knowledge Base
RUN python -m scripts.build_knowledge_base.build_ICD
RUN python -m scripts.build_knowledge_base.build_RXNorm
RUN python -m scripts.build_knowledge_base.build_index

ENTRYPOINT ["python", "-m"]
