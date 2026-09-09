# ingest.py
"""
Build the legal retrieval indices.

Run this script whenever laws.json changes:

    python ingest.py

After successful ingestion, use main.py for queries.
"""

import logging

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

from config import PipelineConfig
from pipeline import LegalRAGPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def build_pipeline() -> LegalRAGPipeline:
    config = PipelineConfig()

    embedding_model = SentenceTransformer(
        "sentence-transformers/la-labse-fen-v1"
    )

    # Baseline reranker. For Persian production use, replace this
    # with a multilingual/Persian-trained reranker.
    reranker_model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    chroma_client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIRECTORY
    )

    return LegalRAGPipeline(
        openai_client=None,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        chroma_client=chroma_client,
        config=config,
    )


def main() -> None:
    pipeline = build_pipeline()

    pipeline.build_indices(
        json_file_path="laws.json"
    )

    print("\n✅ Legal retrieval indices built successfully.")
    print(f"Documents: {len(pipeline.documents)}")
    print(f"BM25 index: {pipeline.bm25_index_path}")
    print(f"Chroma collection: {pipeline.config.CHROMA_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
