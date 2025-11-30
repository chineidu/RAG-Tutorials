# Local Deep Researcher - Package Summary

## Overview

**Local Deep Researcher** is a self-hosted, privacy-focused AI research assistant that performs deep, iterative web research to generate comprehensive, cited reports. It runs entirely on local hardware using Ollama or LM Studio, with optional support for remote LLM providers.

### Key Characteristics

- **Privacy-First**: Runs locally without sending data to external APIs (except for web search)
- **Iterative Research**: Uses a graph-based workflow to progressively refine research through multiple search iterations
- **Flexible Configuration**: Supports multiple LLM providers (Ollama, LMStudio, Remote APIs) and search engines (Tavily, SearXNG, Google, Exa)
- **Structured Output**: Leverages LangGraph for state management and workflow orchestration
- **Production-Ready**: Includes Docker support and LangGraph deployment configuration

---

## Architecture

The package implements a **state-driven research graph** that orchestrates the following workflow:

```
START → generate_query → web_research → summarize_sources → reflect_on_summary
                                                                      ↓
                          finalize_summary ← (if max loops reached) ←┘
                                  ↓                                   ↓
                                 END                    (continue research loop)
```

### Workflow Stages

1. **Query Generation**: Converts the research topic into optimized web search queries
2. **Web Research**: Executes searches using configured search API and fetches results
3. **Summarization**: Synthesizes search results into a running summary
4. **Reflection**: Analyzes summary for knowledge gaps and generates follow-up queries
5. **Routing**: Decides whether to continue research or finalize based on iteration count
6. **Finalization**: Compiles final report with sources and citations

---

## Core Components

### 1. State Management (`graph_states.py`)

Defines three dataclasses for managing research state:

- **`SummaryState`**: Main state container tracking:
  - `research_topic`: User's original research question
  - `search_query`: Current search query being executed
  - `web_research_results`: Accumulated search results
  - `sources_gathered`: List of source URLs and citations
  - `research_loop_count`: Current iteration number
  - `existing_summary`: Running summary of findings

- **`SummaryStateInput`**: Input schema accepting only `research_topic`
- **`SummaryStateOutput`**: Output schema returning only `existing_summary`

### 2. Configuration (`configuration.py`)

The `Configuration` class provides flexible runtime configuration:

```python
class Configuration(BaseModel):
    max_web_research_loops: int = 3  # Number of research iterations
    local_llm: str = "qwen3-4b-instruct-2507"  # LLM model name
    llm_provider: Literal["ollama", "lmstudio", "remote"] = "lmstudio"
    search_api: Literal["tavily", "searxng", "google", "exa"] = "searxng"
    fetch_full_page: bool = True  # Fetch complete page content
    ollama_base_url: str = "http://localhost:11434/"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    strip_thinking_tokens: bool = True  # Remove <think> tags
    use_tool_calling: bool = False  # Tool calling vs JSON mode
```

### 3. Graph Orchestration (`graph.py`)

Implements five node functions that form the research workflow:

#### `generate_query(state, config) -> dict[str, Any]`
Generates optimized search queries using LLM with structured output (tool calling or JSON mode).

**Input**: Research topic and existing summary
**Output**: `{"search_query": "optimized search string"}`

#### `web_research(state, config) -> dict[str, Any]`
Executes web searches using configured search API.

**Supported APIs**:
- **SearXNG**: Self-hosted metasearch engine
- **Tavily**: AI-focused search API
- **Google**: Google Custom Search via Serper
- **Exa**: Neural search engine

**Output**:
```python
{
    "sources_gathered": ["formatted source citations"],
    "research_loop_count": state.research_loop_count + 1,
    "web_research_results": ["formatted search results"]
}
```

#### `summarize_sources(state, config) -> dict[str, Any]`
Creates or extends research summary by integrating new search results.

**Logic**:
- First iteration: Creates new summary from search results
- Subsequent iterations: Merges new results with existing summary
- Maintains coherent narrative flow across iterations

**Output**: `{"existing_summary": "updated summary text"}`

#### `reflect_on_summary(state, config) -> dict[str, Any]`
Analyzes current summary to identify knowledge gaps and generate follow-up queries.

**Output**: `{"search_query": "follow-up query addressing gap"}`

#### `route_research(state, config) -> Literal["finalize_summary", "web_research"]`
Conditional routing based on iteration count:
- If `research_loop_count <= max_web_research_loops`: Continue to `web_research`
- Otherwise: Proceed to `finalize_summary`

#### `finalize_summary(state) -> dict[str, Any]`
Compiles final report with deduplicated sources.

**Output Format**:
```markdown
## Summary:
[Research findings]

### Sources:
- [Source 1]
- [Source 2]
...
```

### 4. Prompt Engineering (`prompts.py`)

Centralized prompt templates for:

- **Query Generation**: `query_writer_prompt`, `json_mode_query_prompt`, `tool_calling_query_prompt`
- **Summarization**: `summarizer_prompt`
- **Reflection**: `reflection_prompt`, `json_mode_reflection_prompt`, `tool_calling_reflection_prompt`

All prompts use structured XML-like tags for clarity and include detailed requirements for output formatting.

### 5. Utilities (`utils.py`)

Helper functions for:

#### Search Integrations
- `searxng_search()`: SearXNG metasearch integration
- `tavily_search()`: Tavily AI search
- `google_search()`: Google Custom Search via Serper
- `exa_search()`: Exa neural search

Each returns: `{"results": [{"title": "", "url": "", "content": "", "raw_content": ""}]}`

#### Content Processing
- `fetch_raw_content(url)`: Retrieves and converts HTML to markdown
- `deduplicate_and_format_sources()`: Deduplicates and formats search results
- `format_sources()`: Formats source citations
- `strip_thinking_tokens()`: Removes LLM reasoning tokens

#### LLM Management
- `get_llm(config)`: Initializes LLM based on provider configuration

### 6. Model Configuration (`model_config.py`)

Enums for supported models:

**LocalModel**: Models for Ollama/LMStudio
- `MISTRAL_7B_INSTRUCT_V0_3_Q4_0`
- `LLAMA3_1_8B`, `LLAMA3_2_3B`
- `QWEN3_4B_INSTRUCT_2507`

**RemoteModel**: Remote API models
- `GPT_5_NANO`, `GEMINI_2_0_FLASH_001`
- `LLAMA_3_3_70B_INSTRUCT`
- Tool-calling optimized models

### 7. Settings Management (`settings.py`)

Pydantic-based settings class loading from environment variables:

```python
class Settings(BaseSettingsConfig):
    # App Configuration
    LLM_PROVIDER: str = "lmstudio"
    MAX_WEB_RESEARCH_LOOPS: int = 3
    SEARCH_API: str = "tavily"
    
    # Local Inference URLs
    OLLAMA_URL: str = "http://localhost:11434/v1"
    LMSTUDIO_URL: str = "http://localhost:1234/v1"
    
    # API Keys (SecretStr for security)
    TAVILY_API_KEY: SecretStr
    SERPER_API_KEY: SecretStr
    EXA_API_KEY: SecretStr
    # ... and more
```

---

## Features

### 1. Multi-Provider LLM Support
- **Ollama**: Local inference with GGUF models
- **LM Studio**: OpenAI-compatible local API
- **Remote APIs**: OpenRouter, Groq, Together AI

### 2. Multiple Search Engines
- **SearXNG**: Privacy-focused metasearch (Docker-deployed)
- **Tavily**: AI-optimized search
- **Google**: Custom search via Serper API
- **Exa**: Neural semantic search

### 3. Structured Output Modes
- **Tool Calling**: Uses LLM native tool/function calling
- **JSON Mode**: Structured JSON output for models without tool support

### 4. Content Fetching Strategies
- **Snippet Mode**: Uses search result snippets only
- **Full Page Mode**: Fetches and processes complete page content

### 5. Token Management
- Configurable token limits per source (default: 1000 tokens)
- Automatic deduplication of search results
- Thinking token removal for reasoning models

### 6. Docker Deployment
Includes `docker-compose.yaml` for:
- SearXNG search engine
- Redis cache
- Qdrant vector store (optional)

### 7. LangGraph Integration
- Defined in `langgraph.json` for deployment
- State checkpointing support
- Streaming support via LangGraph API

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- UV package manager (recommended) or pip
- Ollama or LM Studio (for local inference)
- Docker & Docker Compose (optional, for search engine)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd RAG-Tutorials
   ```

2. **Install dependencies**:
   ```bash
   # Using UV (recommended)
   uv pip install --editable . "langgraph-cli[inmem]"
   
   # Using pip
   pip install -e .
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

4. **Start search engine** (optional, if using SearXNG):
   ```bash
   docker-compose up -d searxng redis
   ```

5. **Start local LLM** (if using Ollama):
   ```bash
   ollama serve
   ollama pull qwen3-4b-instruct-2507
   ```

---

## Usage

### Command Line (LangGraph Dev)

```bash
# Start development server
langgraph dev

# The API will be available at http://localhost:8123
```

### Programmatic Usage

```python
from local_deep_researcher.graph import research_graph
from langchain_core.runnables import RunnableConfig

# Basic usage
result = research_graph.invoke(
    {"research_topic": "What is retrieval augmented generation?"},
    config=RunnableConfig(
        configurable={
            "max_web_research_loops": 2,
            "search_api": "searxng",
            "llm_provider": "lmstudio"
        }
    )
)

print(result["existing_summary"])
```

### Configuration Options

```python
config = RunnableConfig(
    configurable={
        # Research depth
        "max_web_research_loops": 3,  # 1-10 recommended
        
        # LLM settings
        "llm_provider": "lmstudio",  # "ollama" | "lmstudio" | "remote"
        "local_llm": "qwen3-4b-instruct-2507",
        "use_tool_calling": False,  # True for tool-calling models
        "strip_thinking_tokens": True,
        
        # Search settings
        "search_api": "searxng",  # "tavily" | "searxng" | "google" | "exa"
        "fetch_full_page": True,  # Fetch complete pages vs snippets
        
        # API endpoints
        "ollama_base_url": "http://localhost:11434/",
        "lmstudio_base_url": "http://localhost:1234/v1"
    }
)
```

---

## API Reference

### Graph Interface

**Input Schema**:
```python
{
    "research_topic": str  # Required: The research question or topic
}
```

**Output Schema**:
```python
{
    "existing_summary": str  # Markdown-formatted research report with sources
}
```

### Configuration Schema

All configuration fields with types and defaults are defined in `Configuration` class (see Configuration section above).

---

## Development

### Project Structure

```
local_deep_researcher/
├── __init__.py              # Package initialization
├── graph.py                 # Main graph logic and nodes
├── graph_states.py          # State definitions
├── configuration.py         # Configuration management
├── prompts.py              # Prompt templates
├── utils.py                # Helper functions
├── model_config.py         # Model enums
├── settings.py             # Environment settings
├── api/                    # FastAPI application
│   ├── __init__.py
│   └── app.py
└── README.md               # User documentation
```

### Key Design Patterns

1. **State Machine**: LangGraph StateGraph for workflow orchestration
2. **Dependency Injection**: Configuration passed via RunnableConfig
3. **Strategy Pattern**: Pluggable LLM providers and search APIs
4. **Template Method**: Structured output generation abstraction

### Adding New Search Providers

1. Implement search function in `utils.py`:
   ```python
   def new_search(query: str, max_results: int = 3, 
                  fetch_full_page: bool = False) -> dict[str, list[dict[str, Any]]]:
       # Implementation
       return {"results": [...]}
   ```

2. Add to `SearchAPI` enum:
   ```python
   class SearchAPI(str, Enum):
       NEW_API: str = "new_api"
   ```

3. Add case in `web_research()` node:
   ```python
   elif search_api == SearchAPI.NEW_API:
       search_results = new_search(...)
   ```

### Adding New LLM Providers

1. Add provider to `llm_provider` literal in `Configuration`
2. Implement provider case in `get_llm()` in `utils.py`
3. Add required environment variables to `Settings`

---

## Performance Considerations

### Token Usage
- Default: 1000 tokens per source
- Full-page fetching significantly increases tokens
- Consider snippet mode for cost optimization

### Iteration Count
- 2-3 iterations: Quick research (1-2 minutes)
- 4-5 iterations: Deep research (3-5 minutes)
- 6+ iterations: Comprehensive research (5+ minutes)

### Model Selection
- **4B models**: Fast, good for summaries
- **8B models**: Balanced performance
- **70B+ models**: Highest quality, slower

---

## Environment Variables

### Required
- `TAVILY_API_KEY` (if using Tavily)
- `SERPER_API_KEY` (if using Google)
- `EXA_API_KEY` (if using Exa)

### Optional
- `OPENROUTER_API_KEY` (for remote models)
- `LANGCHAIN_API_KEY` (for LangSmith tracing)
- `OLLAMA_URL`, `LMSTUDIO_URL` (custom endpoints)

See `.env.example` for complete list.

---

## Docker Deployment

### Services Included

1. **SearXNG**: Metasearch engine on port 8080
2. **Redis**: Cache for SearXNG
3. **Qdrant**: Vector database (optional, for retrieval)

### Commands

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d searxng

# View logs
docker-compose logs -f searxng

# Stop all services
docker-compose down
```

---

## LangGraph Deployment

The package is configured for LangGraph Cloud/Studio deployment via `langgraph.json`:

```json
{
    "graphs": {
        "local_deep_researcher": "./local_deep_researcher/graph.py:research_graph"
    },
    "python_version": "3.12",
    "dependencies": ["."]
}
```

Deploy with:
```bash
langgraph deploy
```

---

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure Ollama/LMStudio is running
   - Pull the model: `ollama pull <model-name>`

2. **Search API failures**
   - Check API keys in `.env`
   - Verify service availability (SearXNG on http://localhost:8080)
   - Check API rate limits

3. **Timeout errors**
   - Reduce `max_web_research_loops`
   - Disable `fetch_full_page`
   - Use faster LLM model

4. **Out of memory**
   - Use smaller models (3B-4B)
   - Reduce token limit per source
   - Enable `strip_thinking_tokens`

---

## Testing

While the repository includes development dependencies for testing, specific test files are not present in the `local_deep_researcher` package. To test manually:

```python
# Test basic functionality
from local_deep_researcher.graph import research_graph

result = research_graph.invoke({
    "research_topic": "What is Python?"
})

assert "Python" in result["existing_summary"]
assert "Sources:" in result["existing_summary"]
```

---

## Contributing

This package is part of the RAG-Tutorials repository. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## License

This project is licensed under the MIT License. See the `LICENSE` file in the repository root for details.

---

## Related Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangChain Documentation**: https://python.langchain.com/
- **Ollama**: https://ollama.ai/
- **LM Studio**: https://lmstudio.ai/
- **SearXNG**: https://docs.searxng.org/

---

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation in `README.md`
- Review code comments in source files

---

**Last Updated**: January 2025  
**Package Version**: 0.1.0  
**Python Version**: 3.12+
