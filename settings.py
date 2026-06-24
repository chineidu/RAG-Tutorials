import os
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENVIRONMENT = Annotated[
    Literal["development", "production", "sandbox"],
    Field(
        default="development",
        description="The application environment.",
    ),
]


class BaseConfig(BaseSettings):
    """
    Settings class for managing application configuration.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(".env").absolute()),
        env_file_encoding="utf-8",
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    PYTHONWARNINGS: str = "ignore"

    # ======= Server settings =======
    ENV: ENVIRONMENT = "production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @field_validator("PORT", mode="before")
    @classmethod
    def parse_port_fields(cls, v: str | int) -> int:
        """Parses port fields to ensure they are integers."""
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                raise ValueError(f"Invalid port value: {v}") from None

        if isinstance(v, int) and not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")

        return v


class DevelopmentConfig(BaseConfig):
    """Development environment settings."""

    ENV: ENVIRONMENT = "development"
    WORKERS: int = 1
    LIMIT_VALUE: int = 500  # requests per minute
    RELOAD: bool = True
    DEBUG: bool = True

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


class SandboxConfig(BaseConfig):
    """Sandbox environment settings."""

    ENV: ENVIRONMENT = "sandbox"
    WORKERS: int = 1
    LIMIT_VALUE: int = 100  # requests per minute
    RELOAD: bool = False
    DEBUG: bool = False

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


class ProductionConfig(BaseConfig):
    """Production environment settings."""

    ENV: ENVIRONMENT = "production"
    WORKERS: int = 2
    LIMIT_VALUE: int = 250  # requests per minute
    RELOAD: bool = False
    DEBUG: bool = False

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


type ConfigType = DevelopmentConfig | ProductionConfig | SandboxConfig


def refresh_settings() -> ConfigType:
    """Refresh environment variables and return new Settings instance.

    This function reloads environment variables from .env file and creates
    a new Settings instance with the updated values.

    Returns
    -------
    ConfigType
        An instance of the appropriate Settings subclass based on the ENV variable.
    """
    load_dotenv(override=True)
    # Determine environment type; `development` is the default
    env = os.getenv("ENV", "development").lower()

    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "sandbox": SandboxConfig,
        # Map both SANDBOX and STAGING to SandboxConfig
        "staging": SandboxConfig,
    }
    config_cls: type[BaseConfig] = configs.get(env, DevelopmentConfig)

    return config_cls()


app_settings = refresh_settings()
