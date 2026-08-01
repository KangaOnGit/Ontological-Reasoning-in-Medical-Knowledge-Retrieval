from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from src.api.request import TextRequest
from src.inference.pipeline import InferencePipeline
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

CONFIG_NER = load_config("configs/NER.yaml")

app = FastAPI(title="Medical Ontology Retrieval API")

pipeline: InferencePipeline | None = None

def build_pipeline() -> InferencePipeline:
    return InferencePipeline(
        ner_type=CONFIG_NER["default"]["type"],
        ner_model=CONFIG_NER["default"]["model"],
    )

@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = build_pipeline()
    log.info("FastAPI service started and pipeline initialized.")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...)
) -> dict[str, list[dict[str, Any]]]:
    if pipeline is None:
        raise HTTPException(503,
                            "Pipeline not initialized.")

    if Path(file.filename).suffix.lower() != ".txt":
        raise HTTPException(400,
                            "Only .txt files are supported.")

    contents = await file.read()

    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400,
                            "Uploaded file must be UTF-8 encoded.")

    return pipeline.run_text(text)
    
@app.post("/predict/text")
async def predict_text(
    request: TextRequest,
) -> dict[str, list[dict[str, Any]]]:
    if pipeline is None:
        raise HTTPException(503,
                            "Pipeline not initialized.")

    return pipeline.run_text(request.text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False,)

# uvicorn app:app --host 0.0.0.0 --port 8000 --reload