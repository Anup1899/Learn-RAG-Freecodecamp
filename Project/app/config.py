"""
    Centralized configuration for the application.
    Uses pydantic-settings for validated environment variable loading.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

load_dotenv()  # Load environment variables from .env file


class Settings(BaseSettings):

    # LLM Configuration
    openai_api_key: str
    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-4o-mini"

    # Langsmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str
    langchain_project: str = "Freecodecamp-RAG-Project"

    # Application 
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 3600  # Cache time-to-live in seconds
    max_retries: int = 3  # Maximum number of retries for failed requests

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"
    

@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings class.
    This ensures that the settings are loaded only once and reused across the application.
    """
    return Settings()