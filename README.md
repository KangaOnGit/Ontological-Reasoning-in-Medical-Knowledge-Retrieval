# Ontological Reasoning in Medical Knowledge Retrieval for Vietnamese Clinical Notes

An end-to-end biomedical NLP pipeline for extracting medical concepts from Vietnamese clinical text and normalizing them against structured medical ontologies such as ICD-10 and RxNorm.

The system combines LLM-based medical entity extraction, assertion detection, ontology retrieval, and multi-stage ranking to transform noisy clinical narratives into standardized medical concepts.

This project was developed for [Viettel AI Race - Track 2: Medical Ontological Reasoning in Knowledge Retrieval](https://competition.viettel.vn/contests/medical-2026)

The implementation emphasizes modularity and reproducibility, enabling each stage of the pipeline to be developed, evaluated, and replaced independently.

## Highlights

- End-to-end pipeline for extracting medical spans from free-text documents
- Rule-based assertion labeling for extracted entities
- Candidate generation over structured biomedical ontologies
- Hybrid ontology retrieval combining exact alias matching, BM25 sparse retrieval, and FAISS dense retrieval
- Configurable weighted ranking strategy for medical concept normalization
- Modular architecture separating extraction, retrieval, ranking, and submission generation

## System Architecture

<p align="center">
  <img src="images\Ontological_Pipeline.png" width="1000" height="500"><br>
  <b>Figure 1.</b> System Architecture.
</p>

## Problem Statement

Clinical notes contain valuable medical information but are difficult to process automatically due to:

- inconsistent terminology
- abbreviations
- spelling variations
- Vietnamese-English medical code switching
- incomplete or implicit descriptions

This project addresses this challenge by converting free-form clinical text into structured ontology-linked representations.

## Pipeline

The system consists of five major stages:

### 1. Clinical Document Processing

- Parse clinical documents into structured sections
- Preserve section and subsection context
- Generate context-aware chunks for downstream extraction

### 2. Medical Entity Extraction

- Extract medical concepts from clinical text using a locally deployed LLM.
- Supports Vietnamese-English mixed medical terminology

### 3. Assertion Classification

Classify extracted medical mentions according to their clinical context:

- Present conditions
- Negated findings
- Historical mentions
- Uncertain mentions

### 4. Ontology Candidate Retrieval and Ranking

Generate and rank ontology candidates using:

- Exact alias matching
- BM25 sparse retrieval
- FAISS dense retrieval

### 5. Submission Generation

Select final ontology concepts and convert them into the required structured output format.

## Evaluation Metrics

The competition evaluates the system at three levels:

### 1. Concept Mention Extraction (`text_score`)

Concept extraction quality is measured using Word Error Rate (WER) over the extracted text field.

The score is calculated as:

$$
\mathrm{text\_score}=\frac{1}{|test|}\sum_{i\in test}(1-\mathrm{WER}(i))
$$

A lower WER corresponds to a higher text extraction score.

---

### 2. Assertion Classification (`assertions_score`)

Assertion prediction is evaluated using Jaccard similarity between the predicted and ground-truth assertion sets.

For each sample:

$$
J_{\mathrm{assertions}}(i)=\frac{|GT \cap Prediction|}{|GT \cup Prediction|}
$$

Special cases:

- Both ground truth and prediction are empty → score = 1
- Ground truth is empty but prediction is not → score = 0

The final assertion score is the average Jaccard similarity across all test samples.

---

### 3. Ontology Candidate Matching (`candidates_score`)

Candidate normalization is evaluated using Jaccard similarity between predicted ontology candidates and ground-truth candidates.

The final candidate score is weighted by the number of candidate concepts in each sample:

$$\mathrm{candidates\_score} = \frac{\left(\sum_i J_{\mathrm{candidates}}(i)\right)\left(\sum_k (|gt(k)|+1)\right)}{\sum_i \sum_k (|gt(k)|+1)}$$

---

### Final Competition Score

The final score combines all components:

$$
\mathrm{final\_score}=0.3\cdot\mathrm{text\_score}+0.3\cdot\mathrm{assertions\_score}+0.4\cdot\mathrm{candidates\_score}
$$

Because candidate normalization contributes the largest weight, the retrieval and ranking components are designed to maximize ontology matching robustness while maintaining accurate extraction and assertion detection.

## Results

Official benchmark scores are omitted because parts of the competition evaluation protocol, including ontology versions and annotation guidelines, were not publicly disclosed. See Evaluation Limitations for details.

## Retrieval Architecture

The retrieval module follows a multi-stage candidate generation and ranking framework:

1. **Exact Alias Matching**
   - Performs deterministic lookup against curated medical aliases.
   - Provides high-confidence matches when clinical text contains known terminology.

2. **Sparse Retrieval with BM25**
   - Retrieves candidates based on lexical similarity.
   - Handles spelling variations, abbreviations, and partial matches.

3. **Dense Retrieval with FAISS**
   - Uses embedding-based semantic similarity for paraphrases and conceptually related expressions.
   - Helps retrieve concepts when lexical overlap is limited.

4. **Weighted Reciprocal Rank Fusion (Weighted RRF)**

The ranked candidate lists from each retrieval method are combined using **Weighted Reciprocal Rank Fusion (Weighted RRF)**:

$$\mathrm{RRF}(c)= w_e\frac{1}{k+r_{\mathrm{alias}}(c)} + w_b\frac{1}{k+r_{\mathrm{BM25}}(c)} + w_f\frac{1}{k+r_{\mathrm{FAISS}}(c)}$$

where:

* $r_{\mathrm{alias}}(c)$ is the rank of candidate $c$ returned by exact alias matching.
* $r_{\mathrm{BM25}}(c)$ is the rank of candidate $c$ returned by BM25 retrieval.
* $r_{\mathrm{FAISS}}(c)$ is the rank of candidate $c$ returned by FAISS retrieval.
* $w_e$, $w_b$, and $w_f$ are configurable weights assigned to each retrieval method.
* $k$ is the RRF constant (typically 60), which reduces the influence of lower-ranked candidates.

Unlike score-based fusion, Weighted RRF combines candidate rankings rather than raw similarity scores, making the retrieval process robust to differences in scoring scales across retrieval methods.

Exact alias matching provides high precision for known medical terminology, while BM25 and FAISS improve recall for abbreviated, noisy, and semantically similar clinical expressions. The fused ranking leverages the complementary strengths of all three retrieval methods to produce a robust final candidate ranking.

### Models

- NER (self-hosted):
  - Qwen3-8B
  - Qwen3-4B-Instruct-2507
  - GLiNER-multi-v2.1
  - GLiNER-bi-base-v2.0
  
- Text embedding model:
  - SapBERT
  - E5

All models are deployed locally without external API calls, enabling fully offline inference while satisfying the competition constraints. The retrieval pipeline is model-agnostic and can be configured through YAML.

## Technical stack

- Python
- PyTorch and Transformers
- FAISS for vector indexing
- Hugging Face model integration
- YAML-based configuration
- Pandas, NumPy, and PyYAML for data processing and orchestration
- Docker
- FastAPI

## Design Decisions

### LLM-based Medical Entity Extraction


Vietnamese clinical NLP resources remain limited, especially for mixed Vietnamese-English medical notes containing abbreviations, spelling variations, and inconsistent terminology. A self-hosted LLM provides greater robustness to these challenges than traditional sequence-labeling approaches while requiring minimal task-specific training data. The extraction component is designed to prioritize recall, allowing the downstream ontology retrieval and ranking stages to filter and normalize candidate concepts.

### Rule-based Assertion Detection

Due to the limited availability of Vietnamese clinical assertion datasets and pretrained models, a lightweight rule-based classifier was implemented to identify negation, historical, and family-history assertions.

### Section-aware Document Parsing

Clinical documents contain structural information that affects interpretation. Section-aware chunking preserves context and improves extraction reliability.

### Hybrid Ontology Retrieval

Medical terminology has both lexical and semantic variation. Combining exact matching, BM25, and FAISS retrieval balances precision and recall.

## Repository structure

- [src/preprocess](src/preprocess): document parsing and chunking utilities
- [src/NER](src/NER): entity extraction models and inference logic
- [src/assertion](src/assertion): assertion classification logic
- [src/rag](src/rag): knowledge-base construction, encoders, retrievers, and indexing
- [src/postprocess](src/postprocess): span localization and postprocessing
- [src/inference](src/inference): end-to-end inference and submission generation
- [scripts](scripts): runnable scripts for building knowledge bases and executing inference
- [configs](configs): configuration for models, retrieval, and submission settings
- [data](data): curated inputs, ground-truth data, and generated artifacts
- [notebooks](notebooks): exploratory notebooks used during development

## Requirements

This project requires Python 3.10+ and the dependencies listed in [requirements.txt](requirements.txt).

Install them with:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd "Ontological Reasoning in Medical Knowledge Retrieval"
pip install -r requirements.txt
```

### 2. Build the knowledge bases

```bash
python scripts/rag/build_kb/build_ICD.py
python scripts/rag/build_kb/build_RXNorm.py
python scripts/rag/build_index/build_faiss.py
```

These scripts use configuration from [configs/RAG](configs/RAG) and write outputs to [data/KB](data/KB).

### 3. Run inference

```bash
python scripts/submission.py --input_dir "data/Round 1/P2" --output_dir outputs
```

Default model settings and input/output paths are loaded from [configs/NER.yaml](configs/NER.yaml) and [configs/submission.yaml](configs/submission.yaml).

## Docker

A dedicated API Docker image is provided to simplify environment setup and expose the service through FastAPI.

### Build the API image

```bash
docker build -f docker/Dockerfile.api -t medical-ontology-retrieval-api .
```

### Run the API service

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  medical-ontology-retrieval-api
```

This container starts the FastAPI app with Uvicorn and exposes it on port `8000`.

During the image build, the Dockerfile automatically:

* Installs all Python dependencies.
* Builds the ICD-10 and RxNorm knowledge bases.
* Constructs the FAISS index.

### FastAPI serving and deployment

The service entrypoint is the FastAPI app in [app.py](app.py). For local or container-based deployment, start it with:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API provides a health endpoint and a prediction endpoint:

```bash
curl http://localhost:8000/health
```

#### Submit raw text

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=Khách hàng có tiền sử tăng huyết áp và đau ngực."
```

#### Submit a `.txt` file

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.txt"
```

The API accepts either:

* a non-empty `text` form field, or
* a single uploaded `.txt` file

and returns the pipeline inference JSON payload for the selected text input.

### Submission container

A separate submission container is also available for batch CLI runs:

```bash
docker build -f docker/Dockerfile.submission -t medical-ontology-retrieval-submission .
```

It is configured to run the submission pipeline entrypoint:

```bash
python -m scripts.submission
```

## Configuration

Key configuration files include:

- [configs/NER.yaml](configs/NER.yaml): model aliases for the NER component
- [configs/submission.yaml](configs/submission.yaml): default input and output directories
- [configs/RAG/kb_sources/ICD.yaml](configs/RAG/kb_sources/ICD.yaml): ICD knowledge-base settings
- [configs/RAG/kb_sources/RXNorm.yaml](configs/RAG/kb_sources/RXNorm.yaml): RXNorm knowledge-base settings
- [configs/RAG/indexing/faiss_indexing.yaml](configs/RAG/indexing/faiss_indexing.yaml): encoder and FAISS index configuration
- [configs/prompt/span_extraction.jinja](configs/prompt/span_extraction.jinja): prompt template for span extraction

## Output Format

Running the pipeline produces structured per-file submission records and a ZIP archive in the configured output directory.

## Evaluation Limitations

This repository contains the complete end-to-end pipeline developed for Viettel AI Race – Track 2. However, reproducing the official evaluation results is not possible because several aspects of the evaluation protocol were not publicly specified:

- the exact ICD-10 and RxNorm ontology versions
- complete annotation guidelines
- ontology candidate matching rules

Consequently, retrieval scores may differ despite using the same methodology. assertion detection, ontology retrieval, candidate ranking, and inference—are included in this repository.

## Notes

This repository is intended for experimentation and research in biomedical NLP and knowledge-grounded retrieval. It relies on external model downloads and local knowledge-base artifacts, so internet access and sufficient compute resources may be required for full execution.

## Data Sources

### RxNorm

Source:
[NLM RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html)

Release:
Current Prescribable Content Monthly Release - July 6, 2026

MD5:
767678e3b1d6fe358b61c21659f3ef

### ICD-10 Vietnamese Extension

Source:
[TT06/2026/TT-BYT ICD-10 Vietnamese catalog](https://benhviensuoikhoang.com/kham-chua-benh/danh-muc-benh-icd-x/danh-muc-benh-icd-10.html)

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
