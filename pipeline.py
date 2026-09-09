# pipeline.py
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from hazm import Normalizer, WordTokenizer
from rank_bm25 import BM25Okapi

from config import PipelineConfig

logger = logging.getLogger("LegalRAGPipeline")


class LegalRAGPipeline:
    """
    Hybrid Persian legal retrieval pipeline.

    Retrieval flow:
        LLM Intent Mapping
        -> Query Expansion
        -> Dense Retrieval + BM25
        -> RRF Fusion
        -> Cross-Encoder Reranking

    Index construction is intentionally separated from serving:
        ingest.py -> build_indices()
        main.py   -> load_indices() + run()
    """

    def __init__(
        self,
        openai_client: Any,
        embedding_model: Any,
        reranker_model: Any,
        chroma_client: Any,
        config: Optional[PipelineConfig] = None,
    ):
        self.client_openai = openai_client
        self.embedding_model = embedding_model
        self.reranker = reranker_model
        self.chroma_client = chroma_client
        self.config = config or PipelineConfig()

        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()

        self.collection = None
        self.bm25 = None
        self.documents: List[Dict[str, Any]] = []

        self.bm25_index_path = Path(
            getattr(self.config, "BM25_INDEX_PATH", "data/bm25.pkl")
        )
        self.documents_path = Path(
            getattr(self.config, "DOCUMENTS_INDEX_PATH", "data/documents.pkl")
        )

    def _clean_and_tokenize_persian(self, text: str) -> List[str]:
        """Normalize and tokenize Persian text using Hazm."""
        try:
            normalized_text = self.normalizer.normalize(text)
            return self.tokenizer.tokenize(normalized_text)
        except Exception as exc:
            logger.warning("Persian tokenization failed: %s", exc)
            return text.split()

    def build_indices(self, json_file_path: str) -> None:
        """
        Build and persist all retrieval indices.

        This method belongs to the ingestion step and should normally
        be executed through ingest.py, not every time the API starts.
        """
        json_path = Path(json_file_path)
        logger.info("Loading legal corpus from %s", json_path)

        with json_path.open("r", encoding="utf-8") as f:
            laws_data = json.load(f)

        if not isinstance(laws_data, list):
            raise ValueError("laws.json must contain a JSON list.")

        self.documents = []
        texts = []

        for item in laws_data:
            if "text" not in item or "law" not in item:
                raise ValueError("Each law document must contain 'text' and 'law'.")

            text = str(item["text"]).strip()

            document = {
                "text": text,
                "law": str(item["law"]).strip(),
                "article": str(item.get("article", "ماده")).strip(),
            }

            self.documents.append(document)
            texts.append(text)

        if not texts:
            raise ValueError("The legal corpus is empty.")

        logger.info("Encoding %d documents...", len(texts))
        doc_embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
        )

        self.collection = self.chroma_client.get_or_create_collection(
            self.config.CHROMA_COLLECTION_NAME
        )

        all_ids = [str(i) for i in range(len(self.documents))]
        all_metadatas = [
            {"law": doc["law"], "article": doc["article"]}
            for doc in self.documents
        ]

        # upsert makes ingestion safely repeatable.
        self.collection.upsert(
            ids=all_ids,
            documents=texts,
            embeddings=doc_embeddings.tolist(),
            metadatas=all_metadatas,
        )

        logger.info("Building BM25 index...")
        tokenized_corpus = [
            self._clean_and_tokenize_persian(text)
            for text in texts
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents_path.parent.mkdir(parents=True, exist_ok=True)

        with self.bm25_index_path.open("wb") as f:
            pickle.dump(self.bm25, f)

        with self.documents_path.open("wb") as f:
            pickle.dump(self.documents, f)

        logger.info("Indices built and persisted successfully.")

    def load_indices(self) -> None:
        """
        Load previously created Chroma + BM25 indices.

        Run ingest.py first if these files do not exist.
        """
        if not self.bm25_index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {self.bm25_index_path}. "
                "Run ingest.py first."
            )

        if not self.documents_path.exists():
            raise FileNotFoundError(
                f"Documents index not found: {self.documents_path}. "
                "Run ingest.py first."
            )

        try:
            self.collection = self.chroma_client.get_collection(
                self.config.CHROMA_COLLECTION_NAME
            )
        except Exception as exc:
            raise RuntimeError(
                f"Chroma collection "
                f"'{self.config.CHROMA_COLLECTION_NAME}' was not found. "
                "Run ingest.py first."
            ) from exc

        with self.bm25_index_path.open("rb") as f:
            self.bm25 = pickle.load(f)

        with self.documents_path.open("rb") as f:
            self.documents = pickle.load(f)

        logger.info(
            "Indices loaded successfully: %d documents.",
            len(self.documents),
        )

    def legal_intent_mapper(self, user_question: str) -> Dict[str, Any]:
        prompt = f"""
به عنوان یک متخصص سیستم‌های حقوقی ایران، سوال زیر را تحلیل کن و خروجی را دقیقاً
در قالب یک JSON سازگار با فیلدهای زیر برگردان. هیچ متن اضافی ارسال نکن.

سوال کاربر: "{user_question}"

ساختار خروجی:
{{
    "primary_intent": "عنوان یا مفهوم حقوقی دقیق و خلاصه سوال به زبان رسمی حقوقی",
    "must_have_concepts": ["۲ یا ۳ واژه کلیدی بنیادی و عینی موجود در متن قوانین مرتبط"]
}}
"""

        try:
            response = self.client_openai.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.MAPPER_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.error("Intent mapping failed: %s", exc)
            return {
                "primary_intent": user_question,
                "must_have_concepts": [],
            }

    def chroma_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        try:
            q_emb = self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )[0]

            res = self.collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=top_k,
            )

            output = []

            if res and res.get("documents") and res["documents"][0]:
                for idx, text in enumerate(res["documents"][0]):
                    meta = res["metadatas"][0][idx]
                    output.append(
                        {
                            "text": text,
                            "law": meta.get("law", ""),
                            "article": meta.get("article", ""),
                        }
                    )

            return output

        except Exception as exc:
            logger.error("Dense search failed: %s", exc)
            return []

    def bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError("BM25 index is not initialized.")

        try:
            tokenized_query = (
                query
                if isinstance(query, list)
                else self._clean_and_tokenize_persian(query)
            )

            scores = self.bm25.get_scores(tokenized_query)

            if len(scores) == 0:
                return []

            top_idx = np.argsort(scores)[::-1][:top_k]

            return [
                self.documents[i]
                for i in top_idx
                if scores[i] > 0
            ]

        except Exception as exc:
            logger.error("BM25 search failed: %s", exc)
            return []

    @staticmethod
    def _document_uid(doc: Dict[str, Any]) -> str:
        law = doc.get("law", "").strip()
        article = doc.get("article", "").split("(")[0].strip()
        return f"{law}_{article}"

    def cross_encoder_rerank(
        self,
        expanded_query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        try:
            pairs = [
                (expanded_query, doc["text"])
                for doc in candidates
            ]

            scores = self.reranker.predict(pairs)

            results = []

            for doc, score in zip(candidates, scores):
                item = doc.copy()
                item["score"] = float(score)
                results.append(item)

            return sorted(
                results,
                key=lambda x: x["score"],
                reverse=True,
            )

        except Exception as exc:
            logger.error("Reranking failed: %s", exc)
            return candidates

    def run(self, user_question: str) -> List[Dict[str, Any]]:
        if self.collection is None or self.bm25 is None:
            raise ValueError(
                "Pipeline indices are not initialized. "
                "Call load_indices() first."
            )

        if not user_question or not user_question.strip():
            return []

        logger.info("Processing query: %s", user_question)

        intent_map = self.legal_intent_mapper(user_question)

        primary_intent = intent_map.get(
            "primary_intent",
            user_question,
        )

        must_have = intent_map.get(
            "must_have_concepts",
            [],
        )

        if not isinstance(must_have, list):
            must_have = []

        rich_query_1 = (
            f"{user_question} {' '.join(map(str, must_have))}"
        ).strip()

        rich_query_2 = str(primary_intent).strip()

        rich_query_3 = (
            " ".join(map(str, must_have))
            if must_have
            else user_question
        )

        queries = [
            q for q in
            [rich_query_1, rich_query_2, rich_query_3]
            if q.strip()
        ]

        all_docs_info: Dict[str, Dict[str, Any]] = {}

        for query_text in queries:
            dense_res = self.chroma_search(
                query_text,
                top_k=self.config.INITIAL_TOP_K,
            )

            for rank, doc in enumerate(dense_res):
                uid = self._document_uid(doc)

                if uid not in all_docs_info:
                    all_docs_info[uid] = {
                        "doc": doc,
                        "dense_ranks": [],
                        "bm25_ranks": [],
                    }

                all_docs_info[uid]["dense_ranks"].append(rank + 1)

            bm25_res = self.bm25_search(
                query_text,
                top_k=self.config.INITIAL_TOP_K,
            )

            for rank, doc in enumerate(bm25_res):
                uid = self._document_uid(doc)

                if uid not in all_docs_info:
                    all_docs_info[uid] = {
                        "doc": doc,
                        "dense_ranks": [],
                        "bm25_ranks": [],
                    }

                all_docs_info[uid]["bm25_ranks"].append(rank + 1)

        if not all_docs_info:
            return []

        final_ranked_docs = []

        for info in all_docs_info.values():
            best_dense = (
                min(info["dense_ranks"])
                if info["dense_ranks"]
                else float("inf")
            )

            best_bm25 = (
                min(info["bm25_ranks"])
                if info["bm25_ranks"]
                else float("inf")
            )

            dense_part = (
                self.config.W_DENSE /
                (self.config.RRF_K + best_dense)
                if best_dense != float("inf")
                else 0
            )

            bm25_part = (
                self.config.W_BM25 /
                (self.config.RRF_K + best_bm25)
                if best_bm25 != float("inf")
                else 0
            )

            score = dense_part + bm25_part

            final_doc = info["doc"].copy()
            final_doc["rrf_score"] = score

            final_ranked_docs.append(final_doc)

        final_ranked_docs.sort(
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        top_rrf_candidates = final_ranked_docs[
            : self.config.RRF_THRESHOLD_K
        ]

        rerank_query = (
            f"موضوع حقوقی: {primary_intent}\n"
            f"سوال: {user_question}"
        )

        return self.cross_encoder_rerank(
            rerank_query,
            top_rrf_candidates,
        )
