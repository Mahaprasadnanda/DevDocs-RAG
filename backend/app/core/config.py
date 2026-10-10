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
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
