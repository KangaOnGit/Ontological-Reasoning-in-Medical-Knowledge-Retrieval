# Ontological Reasoning in Medical Knowledge Retrieval

An end-to-end biomedical NLP pipeline for extracting medical concepts from Vietnamese clinical text and normalizing them against structured medical ontologies such as ICD-10 and RxNorm.

The system combines LLM-based medical entity extraction, assertion detection, ontology retrieval, and multi-stage ranking to transform noisy clinical narratives into standardized medical concepts.

This project was developed for Viettel AI Race - Track 2: Medical Ontology for Medical Retrieval.

## Highlights

- End-to-end pipeline for extracting medical spans from free-text documents
- Rule-based assertion labeling for extracted entities
- Candidate generation over structured biomedical ontologies
- Hybrid ontology retrieval combining exact alias matching, BM25 sparse retrieval, and FAISS dense retrieval
- Configurable weighted ranking strategy for medical concept normalization
- Modular architecture separating extraction, retrieval, ranking, and submission generation

## System Architecture

```text
Clinical Note
      |
      v
Document Parser
      |
      v
Section-aware Chunking
      |
      v
Medical NER
(LLM-based extraction)
      |
      v
Assertion Detection
      |
      v
+---------------------------+
| Ontology Retrieval        |
|                           |
| Exact Alias Matching      |
| BM25 Sparse Retrieval     |
| FAISS Dense Retrieval     |
+---------------------------+
      |
      v
Weighted Score Fusion
      |
      v
ICD-10 / RxNorm Concept
      |
      v
Submission Generator
```

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

- Extract medical mentions from clinical text using an LLM-based NER component
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

### 5. Concept Normalization and Submission Generation

Select final ontology concepts and convert them into the required structured output format.

## Evaluation Metrics

The competition evaluates the system at three levels:

### 1. Concept Mention Extraction (`text_score`)

Concept extraction quality is measured using Word Error Rate (WER) over the extracted text field.

The score is calculated as:

$$
text\_score =
\frac{1}{|test|}
\sum_{i \in test}(1-WER(i))
$$

A lower WER corresponds to a higher text extraction score.

---

### 2. Assertion Classification (`assertions_score`)

Assertion prediction is evaluated using Jaccard similarity between the predicted and ground-truth assertion sets.

For each sample:

$$
J_{assertions}(i)
=
\frac{|GT \cap Prediction|}
{|GT \cup Prediction|}
$$

Special cases:

- Both ground truth and prediction are empty → score = 1
- Ground truth is empty but prediction is not → score = 0

The final assertion score is the average Jaccard similarity across all test samples.

---

### 3. Ontology Candidate Matching (`candidates_score`)

Candidate normalization is evaluated using Jaccard similarity between predicted ontology candidates and ground-truth candidates.

The final candidate score is weighted by the number of candidate concepts in each sample:

$$
candidates\_score =
\frac{
\sum_i J_{candidates}(i)
\cdot
\sum_k(|ground\_truth(k)|+1)
}
{
\sum_i \sum_k(|ground\_truth(k)|+1)
}
$$

---

### Final Competition Score

The final score combines all components:

$$
final\_score =
0.3 \cdot text\_score
+
0.3 \cdot assertions\_score
+
0.4 \cdot candidates\_score
$$

Because candidate normalization contributes the largest weight, the retrieval and ranking components are designed to maximize ontology matching robustness while maintaining accurate extraction and assertion detection.

## Results

Benchmark results will be added after final evaluation.

| Metric | Score |
|---|---:|
| Text Score | TBD |
| Assertion Score | TBD |
| Candidate Score | TBD |
| Final Score | TBD |

## Retrieval Architecture

The retrieval module follows a multi-stage candidate generation and ranking framework:

1. Exact Alias Matching
   - Performs deterministic lookup against curated medical aliases.
   - Provides high-confidence matches when clinical text contains known terminology.

2. Sparse Retrieval with BM25
   - Retrieves candidates based on lexical similarity.
   - Handles spelling variations, abbreviations, and partial matches.

3. Dense Retrieval with FAISS
   - Uses embedding-based semantic similarity for paraphrases and conceptually related expressions.
   - Helps retrieve concepts when lexical overlap is limited.

4. Weighted Score Fusion

Candidate scores from different retrieval methods are normalized before fusion:

$$
Score(c)=
w_e\hat{S}_{alias}(c)+
w_b\hat{S}_{BM25}(c)+
w_f\hat{S}_{FAISS}(c)
$$

Exact matching improves precision for known medical expressions, while BM25 and FAISS improve recall for noisy, abbreviated, or semantically similar mentions.

where:

- $\hat{S}_{alias}$ represents normalized exact alias matching confidence
- $\hat{S}_{BM25}$ represents normalized lexical similarity
- $\hat{S}_{FAISS}$ represents normalized semantic similarity
- $w_e,w_b,w_f$ are configurable weights

This ranking strategy combines deterministic matching with semantic retrieval, improving robustness for noisy clinical text. 

## Technical stack

- Python
- PyTorch and Transformers
- FAISS for vector indexing
- Hugging Face model integration
- YAML-based configuration
- Pandas, NumPy, and PyYAML for data processing and orchestration

### Models

- LLM-based NER:
  - Qwen3-8B (self-hosted)
  
- Text embedding model:
  - SapBERT

All models are deployed locally without external API calls to satisfy competition constraints.

## Design Decisions

### LLM-based Medical Entity Extraction


Vietnamese clinical NLP resources remain limited, especially for mixed Vietnamese-English medical notes. A self-hosted LLM provides stronger adaptability for noisy real-world clinical language.

### Rule-based Assertion Detection

Due to limited Vietnamese clinical assertion models, a lightweight rule-based classifier was implemented to identify negation and uncertainty patterns.

### Section-aware Document Parsing

Clinical documents contain structural information that affects interpretation. Section-aware chunking preserves context and improves extraction reliability.

### Hybrid Ontology Retrieval

Medical terminology has both lexical and semantic variation. Combining exact matching, BM25, and FAISS retrieval balances precision and recall.

## Repository structure

- [src/preprocess](src/preprocess): document parsing and chunking utilities
- [src/NER](src/NER): entity extraction model and inference logic
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

## Notes

This repository is intended for experimentation and research in biomedical NLP and knowledge-grounded retrieval. It relies on external model downloads and local knowledge-base artifacts, so internet access and sufficient compute resources may be required for full execution.

## Data Sources

### RxNorm

Source:
[NLM RxNorm]

Release:
Current Prescribable Content Monthly Release - July 6, 2026

MD5:
767678e3b1d6fe358b61c21659f3ef

### ICD-10 Vietnamese Extension

Source:
TT06/2026/TT-BYT ICD-10 Vietnamese catalog

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
