# HyRALLM: Hybrid Retrieval-Augmented Generation for Code Summarization

HyRALLM is a comprehensive framework that enhances Large Language Models (LLMs) for source code summarization tasks using a hybrid retrieval mechanism. It intelligently combines dense (semantic) and sparse (BM25) retrievers to fetch the most relevant code-summary pairs from a database, effectively reducing LLM hallucination and improving generation quality.

##  Features

- **Hybrid Retrieval Pipeline**: Fuses Sparse (BM25) and Dense (Contrastive Learning-based) retrievers to maximize retrieval accuracy.
- **Support for Major LLMs**: Out-of-the-box integration for OpenAI, Anthropic Claude, and Gemini APIs via automatic format detection.
- **Multi-Generation Strategies**: Selects the best generation output from multiple localized retrieval prompts utilizing a Token-Level F1 metric.
- **Robust Evaluation Suite**: Built-in support for multiple summarization metrics including ROUGE-L, BLEU-4, and dynamically configurable evaluation loops.
- **Experiment Tracking**: Incremental save functionality to ensure zero data loss during long-running API requests.

##  Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### Configuration

You can configure the project by modifying the `Config` class at the beginning of `run.py`.

- **API Settings**: Set your API Base URL and API Key in `NEWAPI_BASE_URL` and `NEWAPI_API_KEY`.
- **Dataset Path**: Point `DATASET_ROOT` to your dataset folder (e.g., PCSD, JCSD).
- **Retrieval Modes**: Switch between `dense`, `sparse`, or `hybrid` via `RETRIEVAL_METHOD`.

### Running the Pipeline

Execute the main pipeline directly:

```bash
python run.py
```
