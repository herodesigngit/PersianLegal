# Persian Legal RAG — Hybrid Retrieval & Neural Reranking

A production-oriented Persian Legal Retrieval and RAG pipeline combining:

* LLM-based query understanding
* Query expansion
* Dense semantic retrieval
* BM25 lexical retrieval
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder neural reranking
* Persian text normalization and tokenization
* Persistent ChromaDB vector storage

The goal is to retrieve the most legally relevant articles from a Persian legal corpus for a given user question.

Demo:
[🎥 Watch Demo](./video.mp4)
---

## Architecture

```text
                    User Question
                          │
                          ▼
                ┌───────────────────┐
                │  Intent Mapping   │
                │      (LLM)        │
                └─────────┬─────────┘
                          │
                          ▼
                 Query Expansion
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       Dense Retrieval          BM25 Retrieval
        (LA-BSE)                (Lexical Search)
              │                       │
              └───────────┬───────────┘
                          ▼
                Reciprocal Rank
                   Fusion
                     (RRF)
                          │
                          ▼
                 Candidate Documents
                          │
                          ▼
                Cross-Encoder Reranker
                          │
                          ▼
                  Final Ranking
```

---

## Why Hybrid Retrieval?

Legal questions often contain both:

1. **Semantic meaning**
2. **Exact legal terminology**

Dense retrieval is effective when the user expresses a legal concept using different wording.

BM25 is useful when exact terms, article numbers, legal phrases, or specific terminology matter.

Therefore, this project combines both retrieval strategies instead of relying on vector search alone.

---

## Main Components

### 1. Persian Text Processing

Persian queries are normalized and tokenized using Hazm before lexical retrieval.

```python
Normalizer()
WordTokenizer()
```

This helps reduce inconsistencies caused by Persian character and word normalization.

---

### 2. Dense Retrieval

Semantic embeddings are generated using:

```text
sentence-transformers/la-labse-fen-v1
```

The embeddings are stored in persistent ChromaDB.

Dense retrieval allows semantically similar legal articles to be retrieved even when the query and document use different wording.

---

### 3. BM25 Retrieval

BM25 provides lexical retrieval over the legal corpus.

This is particularly useful for:

* legal terminology
* exact phrases
* article-specific vocabulary
* rare keywords

---

### 4. Query Expansion

The original user query is transformed into multiple search formulations.

This increases the probability that both lexical and semantic retrieval systems find relevant legal articles.

---

### 5. Reciprocal Rank Fusion

Results from dense retrieval and BM25 are merged using Reciprocal Rank Fusion.

Conceptually:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

This allows multiple retrieval systems to contribute to the final candidate ranking.

---

### 6. Neural Reranking

The retrieved candidates are passed through a Cross-Encoder to obtain a more fine-grained relevance score.

The reranker operates only on a small candidate set, making the architecture more efficient than applying a Cross-Encoder to the entire corpus.

> Note: The current baseline reranker is an English MS MARCO model. It is included as an experimental baseline and should be replaced with a multilingual/Persian legal reranker for a Persian-focused production system.

---

## Project Structure

```text
legal-rag/
│
├── main.py
├── pipeline.py
├── config.py
├── ingest.py
├── laws.json
│
├── evaluation/
│   ├── eval_dataset.json
│   └── evaluate.py
│
├── tests/
│   ├── test_pipeline.py
│   └── test_retrieval.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd legal-rag

python -m venv .venv
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

Never commit the `.env` file.

Use `.env.example` as the template.

---

## Building the Index

Index construction should be performed separately from query serving.

```bash
python ingest.py
```

The ingestion pipeline:

1. Loads the legal corpus
2. Normalizes documents
3. Generates embeddings
4. Stores embeddings in ChromaDB
5. Builds the BM25 index

---

## Running a Query

```bash
python main.py
```

Example:

```text
اگر در یک دعوای حقوقی خواهان بخواهد مبلغی که ابتدا مطالبه کرده بود
را کم کند، تا چه زمانی این حق را دارد؟
```

The system retrieves and ranks the most relevant legal articles.

Example output:

```text
[1] قانون: ...
    ماده: ...
    score: ...

[2] قانون: ...
    ماده: ...
    score: ...

[3] قانون: ...
    ماده: ...
    score: ...
```

---

# Evaluation

A major goal of this project is to evaluate retrieval quality rather than relying only on subjective inspection.

The evaluation dataset contains:

```text
question
relevant_articles
```

Example:

```json
{
  "question": "خواهان تا چه زمانی می‌تواند میزان خواسته خود را کاهش دهد؟",
  "relevant_articles": [
    "law_name_article_123"
  ]
}
```

The following retrieval configurations should be compared:

```text
1. BM25
2. Dense Retrieval
3. Hybrid Retrieval
4. Hybrid + Reranker
```

Recommended metrics:

* Recall@5
* Recall@10
* MRR@5
* MRR@10
* NDCG@10

The objective is to demonstrate whether each component actually improves retrieval performance.

---

## Evaluation Philosophy

The project should not claim that the reranker or hybrid architecture is better without experimental evidence.

For example:

```text
Method                  Recall@10
----------------------------------
BM25                     0.xx
Dense                    0.xx
Hybrid                   0.xx
Hybrid + Reranker        0.xx
```

This makes the project reproducible and demonstrates an engineering/research mindset.

---

# Tests

Run:

```bash
pytest
```

With coverage:

```bash
pytest --cov=. --cov-report=term-missing
```

Tests should cover:

* Persian normalization
* tokenization
* BM25 retrieval
* result merging
* RRF ranking
* duplicate document handling
* pipeline output structure
* empty/invalid input handling

---

# Limitations

This project is a retrieval-focused legal RAG prototype.

Current limitations include:

* The reranker baseline is not Persian-specific.
* Retrieval quality depends on corpus quality.
* Query expansion uses an LLM and therefore introduces additional latency and cost.
* The current system does not provide a full legal answer-generation layer.
* Retrieval quality requires a manually labeled evaluation dataset.
* Production deployment requires additional monitoring, caching, authentication, logging and API-level controls.

---

# Future Improvements

Potential improvements include:

### Retrieval

* Persian/multilingual legal reranker
* BGE-M3 or another multilingual embedding model
* Metadata filtering
* Parent-child document retrieval
* Hierarchical retrieval
* Legal citation-aware retrieval

### Evaluation

* 200–500 labeled legal questions
* Hard negative mining
* Retrieval ablation studies
* Recall@K / MRR / NDCG dashboards

### RAG

* Citation-aware answer generation
* Citation verification
* Hallucination detection
* LLM-as-a-judge evaluation

### Production

* FastAPI API
* Docker
* Redis caching
* asynchronous inference
* structured logging
* monitoring
* CI/CD

---

# Key Engineering Decisions

### Hybrid Retrieval

Combining BM25 and dense retrieval reduces dependence on a single retrieval paradigm.

### Candidate Reranking

The Cross-Encoder is applied after initial retrieval to balance retrieval quality and computational cost.

### Dependency Injection

The pipeline receives its clients and models through its constructor, making the architecture easier to test and replace.

### Persistent Vector Database

ChromaDB persistence allows embeddings to survive application restarts.

---

# Disclaimer

This project is a technical demonstration of information retrieval and RAG techniques.

It is not a substitute for professional legal advice.
