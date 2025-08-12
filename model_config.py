from enum import Enum


class LocalModel(str, Enum):
    """
    Local LLMs.
    """

    def __str__(self) -> str:
        """
        Return the string representation of the model name.
        """
        return str(self.value)

    MISTRAL_7B_INSTRUCT_V0_3_Q4_0 = "mistral:7b-instruct-v0.3-q4_0"
    LLAMA3_1_8B = "llama3.1:8b"
    LLAMA3_2_3B = "llama3.2:3b"
    QWEN3_4B = "qwen3:4b"
    MXBAI_EMBED_LARGE = "mxbai-embed-large:latest"


class RemoteModel(str, Enum):
    """Remote LLMs."""

    def __str__(self) -> str:
        """
        Return the string representation of the model name.
        """
        return str(self.value)

    GEMINI_2_0_FLASH_001 = "google/gemini-2.0-flash-001"
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    KIMI_K2 = "moonshotai/kimi-k2"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"
    GPT_5_NANO = "openai/gpt-5-nano"
