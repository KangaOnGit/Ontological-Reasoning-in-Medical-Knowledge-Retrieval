from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.inference.pipeline import InferencePipeline
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

CONFIG_NER = load_config("configs/NER.yaml")
CONFIG_SUBMISSION = load_config("configs/submission.yaml")

app = FastAPI(title="Medical Ontology Retrieval API")

pipeline: InferencePipeline | None = None


def build_pipeline() -> InferencePipeline:
    return InferencePipeline(
        ner_model=CONFIG_NER["model"]["default"]
        )

@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = build_pipeline()
    log.info("FastAPI service started and pipeline initialized.")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def run_prediction(
    text: str | None = None,
    file: UploadFile | None = None
    ) -> dict[str, list[dict[str, Any]]]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if file is not None and text is not None and text.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide either text or a file, not both."
        )

    if file is not None:
        if (not file.filename
            or Path(file.filename).suffix.lower() != ".txt"):
            raise HTTPException(status_code=400, detail="Only .txt uploads are supported")

        contents = await file.read()
        if not contents.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / file.filename
            input_path.write_bytes(contents)

            try:
                results = pipeline.run_file(input_path)
                return results
            
            except Exception:
                log.exception("Prediction failed")
                raise HTTPException(status_code=500,
                                    detail="Internal server error.")

    if text is None or not text.strip():
        raise HTTPException(status_code=400, detail="Either a .txt file or non-empty text input must be provided")
    
    try:
        results = pipeline.run_text(text)
        return results
    
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500,
                            detail="Internal server error.")


@app.post("/predict")
async def predict(
    text: str = Form(default=""),
    file: UploadFile | None = File(default=None),
) -> dict[str, list[dict[str, Any]]]:
    return await run_prediction(text=text or None, file=file)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

# uvicorn app:app --host 0.0.0.0 --port 8000