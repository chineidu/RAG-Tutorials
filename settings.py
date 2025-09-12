from pathlib import Path

from dotenv import load_dotenv  # type: ignore
from pydantic import SecretStr  # type: ignore
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class BaseSettingsConfig(BaseSettings):
    """Base configuration class for settings.

    This class extends BaseSettings to provide common configuration options
    for environment variable loading and processing.

    Attributes
    ----------
    model_config : SettingsConfigDict
        Configuration dictionary for the settings model specifying env file location,
        encoding and other processing options.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(".env").absolute()),
        env_file_encoding="utf-8",
        from_attributes=True,
        populate_by_name=True,
    )


class Settings(BaseSettingsConfig):
    """Application settings class containing database and other credentials."""

    # ===== APP CONFIGURATION =====
    LLM_PROVIDER: str = "lmstudio"  # Options: 'ollama', 'lmstudio', 'remote'
    MAX_WEB_RESEARCH_LOOPS: int = 3
    SEARCH_API: str = "tavily"  # Options: 'tavily', 'serper'
    FETCH_FULL_PAGE: bool = True

    # ===== LOCAL INFERENCE =====
    # OLLAMA
    OLLAMA_API_KEY: SecretStr = SecretStr("")
    OLLAMA_URL: str = "http://localhost:11434/v1"

    # LMSTUDIO
    LMSTUDIO_API_KEY: SecretStr = SecretStr("")
    LMSTUDIO_URL: str = "http://localhost:1234/v1"

    # ===== REMOTE INFERENCE =====
    # GROQ
    GROQ_API_KEY: SecretStr = SecretStr("")

    # TOGETHER AI
    TOGETHER_API_KEY: SecretStr = SecretStr("")
    TOGETHER_API_URL: str = "https://api.together.xyz/v1"

    # OPENROUTER
    OPENROUTER_API_KEY: SecretStr = SecretStr("")
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"

    # MISTRAL AI
    MISTRAL_API_KEY: SecretStr = SecretStr("")

    # HUGGINGFACE
    HUGGINGFACE_API_KEY: SecretStr = SecretStr("")

    # LLAMA CLOUD
    LLAMA_CLOUD_API_KEY: SecretStr = SecretStr("")

    # TAVILY
    TAVILY_API_KEY: SecretStr = SecretStr("")

    # GOOGLE SERP
    SERPER_API_KEY: SecretStr = SecretStr("")

    # EXA
    EXA_API_KEY: SecretStr = SecretStr("")

    # ===== OBSERVABILITY =====
    # LANGFUSE
    LANGFUSE_SECRET_KEY: SecretStr = SecretStr("")
    LANGFUSE_PUBLIC_KEY: SecretStr = SecretStr("")
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # LANGFUSE
    LANGCHAIN_API_KEY: SecretStr = SecretStr("")
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "RAG-Tutorials"

    # ===== VECTOR STORE =====
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: SecretStr = SecretStr("")


def refresh_settings() -> Settings:
    """Refresh environment variables and return new Settings instance.

    This function reloads environment variables from .env file and creates
    a new Settings instance with the updated values.

    Returns
    -------
    Settings
        A new Settings instance with refreshed environment variables
    """
    load_dotenv(override=True)
    return Settings()
