# Ontological Reasoning in Medical Knowledge Retrieval

This repository contains a research-oriented pipeline for building and retrieving medical knowledge from structured ontologies such as ICD-10 and RXNorm. The project is intended to support ontology-aware retrieval and reasoning over medical concepts, with a focus on preparing curated knowledge bases and indexing them for semantic search.

> This README is still being expanded as the project evolves. The structure and scripts are already in place, but some parts of the workflow are still being refined.

## Project goal

The main objective is to:

- parse and preprocess medical knowledge sources,
- build curated knowledge-base tables for clinical terminology,
- embed medical terms into vector space,
- create FAISS indexes for efficient retrieval,
- support downstream retrieval-augmented reasoning or QA workflows.

## Repository layout

- [scripts/build_knowledge_base](scripts/build_knowledge_base) contains the knowledge-base builders for ICD, RXNorm, and vector index generation.
- [src/preprocess](src/preprocess) contains parsing and chunking utilities for preparing text and structured data.
- [src/rag](src/rag) contains retrieval and knowledge-base related components.
- [src/models](src/models), [src/postprocess](src/postprocess), and [src/rules](src/rules) hold model, search, and rule-based logic.
- [configs](configs) stores YAML configuration files for the LLM and knowledge-base pipelines.
- [data](data) contains curated input files, ground-truth examples, and generated knowledge-base artifacts.
- [notebooks](notebooks) includes exploratory notebooks used during development.

## Requirements

The project expects Python 3.11+ and the dependencies listed in [requirements.txt](requirements.txt).

Install them with:

```bash
pip install -r requirements.txt
```

Optional environment variables:

- HF_TOKEN: used by Hugging Face integrations when downloading models or artifacts.

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the knowledge bases

The repository includes scripts to build ICD and RXNorm knowledge bases and then generate vector indexes.

```bash
python -m python -m scripts.build_rag.build_kb.build_ICD
python -m python -m scripts.build_rag.build_kb..build_RXNorm
python -m python -m scripts.build_rag.build_index.build_faiss_index
```

These commands rely on the YAML files in [configs/KB](configs/KB) and write outputs under [data/KB](data/KB).

### 3. Run with Docker

A Docker image is also provided via [Dockerfile](Dockerfile):

```bash
docker build -t medical-kg-retrieval .
```

The container image runs the knowledge-base build pipeline automatically.

## Configuration

Configuration files live under [configs](configs):

- [configs/llm.yaml](configs/llm.yaml): placeholder for LLM-related configuration.
- [configs/KB/ICD.yaml](configs/KB/ICD.yaml): ICD knowledge-base settings.
- [configs/KB/RXNorm.yaml](configs/KB/RXNorm.yaml): RXNorm knowledge-base settings.
- [configs/KB/indexing.yaml](configs/KB/indexing.yaml): embedding model and index-output configuration.

## Current status

The repository already includes:

- preprocessing and parsing utilities,
- ICD and RXNorm knowledge-base builders,
- FAISS index construction logic,
- configuration and data folders.

Planned work includes refining the retrieval pipeline, integrating the query flow more fully, and documenting the end-to-end inference workflow.

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
