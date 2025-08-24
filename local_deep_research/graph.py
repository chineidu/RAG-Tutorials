import json
from typing import Any, Type, TypeVar

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .configuration import Configuration, SearchAPI
from .graph_states import SummaryState
from .model_config import LocalModel
from .settings import refresh_settings
from .utils import strip_thinking_tokens

settings = refresh_settings()
T = TypeVar("T", bound=BaseModel)


def get_llm(configurable: Configuration) -> ChatOpenAI:
    """Helper function to initialize LLM based on configuration.

    Uses JSON mode if use_tool_calling is False, otherwise regular mode for tool calling.

    Args:
        configurable: Configuration object containing LLM settings

    Returns:
        Configured LLM instance
    """
    if configurable.llm_provider == "lmstudio":
        return ChatOpenAI(
            base_url=settings.LMSTUDIO_URL,  # type: ignore
            model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
            temperature=0,
        )
    # Default to Ollama
    return ChatOpenAI(
        base_url=settings.OLLAMA_URL,  # type: ignore
        model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
        temperature=0,
    )


def generate_search_query_with_structured_output(
    configurable: Configuration,
    messages: list,
    tool_class: Type[T],
    fallback_query: str,
    tool_query_field: str,
    json_query_field: str,
) -> dict[str, Any]:
    """
    Generate a search query from LLM output using either tool-calling or JSON output.

    Parameters
    ----------
    configurable : Configuration
        Configuration object with LLM and mode settings.
    messages : list
        List of messages to send to the LLM.
    tool_class : Type[T]
        Tool class used for tool-calling mode (should be a pydantic BaseModel subclass).
    fallback_query : str
        Fallback search query to return if extraction fails.
    tool_query_field : str
        Field name in the tool call arguments that contains the query.
    json_query_field : str
        Field name in the JSON response that contains the query.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the key "search_query" with the extracted or fallback query.

    Notes
    -----
    Local LLMs often perform better with JSON mode except when tool calling has an
    optimized prompt. The function attempts tool-calling extraction first when
    configurable.use_tool_calling is True, otherwise it parses JSON content.
    """
    if configurable.use_tool_calling:
        llm: ChatOpenAI = get_llm(configurable).bind_tools([tool_class])  # type: ignore
        result = llm.invoke(messages)

        if not result.tool_calls:  # type: ignore
            # If no tool calls were made
            return {"search_query": fallback_query}

        try:
            tool_data = result.tool_calls[0]["args"]  # type: ignore
            search_query = tool_data.get(tool_query_field)
            return {"search_query": search_query}

        except (IndexError, KeyError):
            return {"search_query": fallback_query}

    else:
        # Use JSON mode
        llm = get_llm(configurable)
        result = llm.invoke(messages)
        print(f"result: {result}")
        content: str = result.content # type: ignore

        try:
            parsed_json: dict[str, Any] = json.loads(content)
            search_query = parsed_json.get(json_query_field)
            if not search_query:
                return {"search_query": fallback_query}
            return {"search_query": search_query}

        except (json.JSONDecodeError, KeyError):
            if configurable.strip_thinking_tokens:
                content = strip_thinking_tokens(content)
            return {"search_query": fallback_query}


def generate_query(state: SummaryState, config: RunnableConfig) -> dict[str, Any]:
    """
    Generate a search query for web search using LLM output.

    Parameters
    ----------
    state : SummaryState
        The current summary state containing the research topic.
    config : RunnableConfig
        Configuration for the LLM and query generation.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the key "search_query" with the generated or fallback query.

    Notes
    -----
    The function formats a prompt using the current date and research topic, then uses either
    tool-calling or JSON mode to extract the search query from the LLM output.
    """

    # Format the prompt
    current_date: str = get_current_date()
    formatted_prompt = query_writer_prompt.format(current_date=current_date, research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)

    @tool
    class Query(BaseModel):
        """
        This tool is used to generate a query for web search.
        """

        query: str = Field(description="The actual search query string")
        rationale: str = Field(description="Brief explanation of why this query is relevant")

    messages = [
        SystemMessage(
            content=formatted_prompt
            + (tool_calling_query_prompt if configurable.use_tool_calling else json_mode_query_prompt)
        ),
        HumanMessage(content="Generate a query for web search:"),
    ]

    return generate_search_query_with_structured_output(
        configurable=configurable,
        messages=messages,
        tool_class=Query,
        fallback_query=f"Tell me more about {state.research_topic}",
        tool_query_field="query",
        json_query_field="query",
    )


# Select the search type from the config, run the query and return the formatted search results
def web_research(state: SummaryState, config: RunnableConfig) -> dict[str, Any]:
    configurable = Configuration.from_runnable_config(config)
    search_api = configurable.search_api

    # Search the web
    if search_api == SearchAPI.SEARXNG:
        search_results: dict[str, list[dict[str, Any]]] = searxng_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str: str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )

    elif search_api == SearchAPI.EXA:
        search_results = exa_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == SearchAPI.TAVILY:
        search_results = tavily_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == SearchAPI.GOOGLE:
        search_results = google_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str],
    }


# - if an existing_summary exist, combine it with the most recent summary and return the combined summary
# - if no existing_summary exists, return the most recent summary
# -create an instance of the llm and call it with the combined summary


def summarize_sources(state: SummaryState, config: RunnableConfig) -> Any:
    existing_summary = state.existing_summary
    most_recent_web_search = state.web_research_results[-1]

    # Human_message
    if existing_summary:
        human_msg_content: str = f"""
        <existing_summary>\n {existing_summary} \n </existing_summary>
        <new_context>\n {most_recent_web_search} \n </new_context>
        Update the existing_summary with the new context of this topic: <user_input> {state.research_topic} in <user_input>
        """
    else:
        human_msg_content = f"""
        <context>\n {most_recent_web_search} \n </context>
        Using this context: \n<user_input> {state.research_topic} in <user_input>
        create a summary
        """

    configurable = configuration.from_runnable_config(config)
    if configurable.llm_provider == "llmstudio":
        llm: ChatOpenAI = ChatOpenAI(
            base_url=settings.LMSTUDIO_URL,  # type: ignore
            model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            base_url=settings.OLLAMA_URL,  # type: ignore
            model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
            temperature=0,
        )

    result = llm.invoke([SystemMessage(content=summarizer_prompt), HumanMessage(content=human_msg_content)])
    existing_summary: str = result.content
    if configurable.strip_thinking_tokens:
        existing_summary = strip_thinking_tokens(existing_summary)

    return {"existing_summary": existing_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig) -> dict[str, Any]:
    # Get the required objects(llm, prompts, etc)
    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = reflection_prompt.format(research_topic=state.research_topic)

    @tool
    class FollowUpQuery(BaseModel):
        follow_up_query: str = Field(description="A specific question to address the knowledge gap")
        knowledge_gap: str = Field(description="Describe what information is missing or needs clarification")

    messages: list[AnyMessage] = [
        SystemMessage(content=formatted_prompt) + tool_calling_reflection_prompt
        if configurable.use_tool_calling
        else json_mode_reflection_prompt,
    ]
    return generate_search_query_with_structured_output(
        configurable=configurable,
        messages=messages,
        tool_class=FollowUpQuery,
        fallback_query=f"Tell me more about {state.research_topic}",
        tool_query_field="follow_up_query",
        json_query_field="follow_up_query",
    )
