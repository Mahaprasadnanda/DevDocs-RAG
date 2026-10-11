from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "DevDocs RAG API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # Ingestion Settings
    INGESTION_REQUEST_TIMEOUT: float = 10.0
    INGESTION_MAX_RESPONSE_BYTES: int = 5_000_000  # 5 MB
    INGESTION_OUTPUT_DIR: str = "data/ingested"
    INGESTION_USER_AGENT: str = "DevDocs-RAG-Ingestion-Bot/1.0"
    
    # Chunking Settings
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    CHUNKING_OUTPUT_DIR: str = "data/chunks"
    
    # Vector Index Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    INDEX_OUTPUT_DIR: str = "data/index"
    RETRIEVAL_MIN_SCORE: float = 0.30
    
    # Groq LLM Settings
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com"
    GROQ_TIMEOUT_SECONDS: float = 30.0
    RAG_CONTEXT_MAX_CHARS: int = 4000
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
