from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # OLLAMA
    OLLAMA_API_KEY: SecretStr = SecretStr("")
    OLLAMA_URL: str = "http://localhost:11434/v1"
    # GROQ
    GROQ_API_KEY: SecretStr = SecretStr("")

    # TOGETHER AI
    TOGETHER_API_KEY: SecretStr = SecretStr("")

    # LANGFUSE
    LANGFUSE_SECRET_KEY: SecretStr = SecretStr("")
    LANGFUSE_PUBLIC_KEY: SecretStr = SecretStr("")
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # OPENROUTER
    OPENROUTER_API_KEY: SecretStr = SecretStr("")
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"

    # MISTRAL AI
    MISTRAL_API_KEY: SecretStr = SecretStr("")

    # HUGGINGFACE
    HUGGINGFACE_API_KEY: SecretStr = SecretStr("")


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
