from __future__ import annotations

import logging
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

async def run_prediction(
    text: str | None = None,
    file: UploadFile | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized.",
        )

    if file is not None and text is not None and text.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide either text or a file, not both.",
        )

    try:
        if file is not None:
            if (
                not file.filename
                or Path(file.filename).suffix.lower() != ".txt"
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Only .txt uploads are supported.",
                )

            contents = await file.read()

            if not contents.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty.",
                )

            try:
                text = contents.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file must be UTF-8 encoded.",
                )

            return pipeline.run_text(text)

        if text is None or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Either a .txt file or non-empty text input must be provided.",
            )

        return pipeline.run_text(text)

    except HTTPException:
        raise

    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


@app.post("/predict")
async def predict(
    text: str = Form(default=""),
    file: UploadFile | None = File(default=None),
) -> dict[str, list[dict[str, Any]]]:
    return await run_prediction(
        text=text or None,
        file=file,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False,)

# uvicorn app:app --host 0.0.0.0 --port 8000