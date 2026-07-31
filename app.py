from __future__ import annotations

import logging
import os
import tempfile
from argparse import Namespace
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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

_pipeline: InferencePipeline | None = None


def build_pipeline(input_dir: str) -> InferencePipeline:
    ner_model_name = os.getenv("NER_MODEL", "Qwen3-8B")
    return InferencePipeline(
        Namespace(
            input_dir=input_dir,
            ner_model=ner_model_name,
            prompt_path="configs/prompt/span_extraction.jinja",
            max_new_tokens=1024,
            repetition_penalty=1.05,
            output_dir=None,
        )
    )


@app.on_event("startup")
def startup() -> None:
    global _pipeline

    default_input_dir = CONFIG_SUBMISSION["data"]["eval"]["path"]
    _pipeline = build_pipeline(default_input_dir)
    log.info("FastAPI service started and pipeline initialized.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _run_prediction(text: str | None = None, file: UploadFile | None = None) -> JSONResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if file is not None:
        if not file.filename or not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt uploads are supported")

        contents = await file.read()
        if not contents.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / file.filename
            input_path.write_bytes(contents)

            try:
                results = _pipeline.run_inference(input_path)
                return JSONResponse(content=results)
            except Exception as exc:
                log.exception("Prediction failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

    if text is None or not text.strip():
        raise HTTPException(status_code=400, detail="Either a .txt file or non-empty text input must be provided")

    try:
        results = _pipeline.run_inference(text)
        return JSONResponse(content=results)
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict")
async def predict(
    text: str = Form(default=""),
    file: UploadFile | None = File(default=None),
) -> JSONResponse:
    return await _run_prediction(text=text or None, file=file)


@app.post("/predict-from-text")
async def predict_from_text(text: str = Form(...)) -> JSONResponse:
    return await _run_prediction(text=text, file=None)


@app.post("/predict-from-file")
async def predict_from_file(file: UploadFile = File(...)) -> JSONResponse:
    return await _run_prediction(text=None, file=file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

# uvicorn app:app --host 0.0.0.0 --port 8000