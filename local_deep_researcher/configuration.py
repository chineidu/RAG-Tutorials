import os
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from local_deep_researcher.model_config import LocalModel


class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default=LocalModel.QWEN3_4B_INSTRUCT_2507,
        title="LLM Model Name",
        description="Name of the LLM model to use",
    )
    llm_provider: Literal["ollama", "lmstudio", "remote"] = Field(
        default="lmstudio",
        title="LLM Provider",
        description="Provider for the LLM (Ollama, LMStudio or Remote API)",
    )
    search_api: Literal["tavily", "searxng", "google", "exa"] = Field(
        default="searxng", title="Search API", description="Web search API to use"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """
        Create a Configuration instance from a RunnableConfig.

        Parameters
        ----------
        config : RunnableConfig | None, optional
            A mapping-like object that may contain a "configurable" key whose value is a
            dictionary of configuration overrides. If None or the "configurable" key is
            absent, environment variables will be consulted for values.

        Returns
        -------
        Configuration
            A new Configuration instance populated with values sourced from environment
            variables `config["configurable"]`. Only keys that correspond to fields in
            `Configuration.model_fields` are considered; any entries with value ``None``
            are filtered out before instantiation.
        """
        configurable: dict[str, Any] = config["configurable"] if config and "configurable" in config else {}

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name)) for name in cls.model_fields.keys()
        }

        # Filter out None values
        values: dict[str, Any] = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
