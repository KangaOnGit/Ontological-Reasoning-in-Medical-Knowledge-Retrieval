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

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build Knowledge Base
RUN python -m scripts.rag.build_kb.build_ICD \
    && python -m scripts.rag.build_kb.build_RXNorm \
    && python -m scripts.rag.build_index.build_faiss

ENTRYPOINT ["python", "-m", "scripts.submission"]

CMD ["--output_dir", "outputs"]