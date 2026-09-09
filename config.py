# config.py
import os

class PipelineConfig:
    # هایپرپارامترهای جستجو
    INITIAL_TOP_K: int = 25
    RRF_THRESHOLD_K: int = 20
    RRF_K: int = 60
    W_DENSE: float = 0.7
    W_BM25: float = 0.3
    
    # تنظیمات مدل‌ها
    OPENAI_MODEL: str = "gpt-4o-mini"
    MAPPER_TEMPERATURE: float = 0.15
    
    # تنظیمات ChromaDB (استفاده از PersistentClient برای ذخیره روی هارد)
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "laws_clean")