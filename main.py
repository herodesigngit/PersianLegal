# main.py
"""
Query the already-built legal retrieval system.

Important:
    Run `python ingest.py` once before running this script,
    or whenever laws.json is updated.
"""

import logging

from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

from config import PipelineConfig
from pipeline import LegalRAGPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def init_pipeline() -> LegalRAGPipeline:
    config = PipelineConfig()

    openai_client = OpenAI()

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

    pipeline = LegalRAGPipeline(
        openai_client=openai_client,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        chroma_client=chroma_client,
        config=config,
    )

    # No expensive embedding/indexing happens here.
    pipeline.load_indices()

    return pipeline


if __name__ == "__main__":
    rag_system = init_pipeline()

    query = (
        "اگر در یک دعوای حقوقی خواهان بخواهد مبلغی که ابتدا "
        "مطالبه کرده بود را کم کند، تا چه زمانی این حق را دارد؟"
    )

    results = rag_system.run(query)

    print("\n=== Final Ranked Results ===")

    for idx, doc in enumerate(results[:3]):
        print(
            f"[{idx + 1}] "
            f"قانون: {doc['law']} - "
            f"ماده: {doc['article']} | "
            f"Score: {doc.get('score', 0):.4f}"
        )
