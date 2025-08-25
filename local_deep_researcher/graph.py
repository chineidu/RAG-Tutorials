import json
from typing import Any, Literal, Type, TypeVar

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from local_deep_researcher.configuration import Configuration
from local_deep_researcher.graph_states import SummaryState, SummaryStateInput, SummaryStateOutput
from local_deep_researcher.model_config import LocalModel
from local_deep_researcher.prompts import (
    get_current_date,
    json_mode_query_prompt,
    json_mode_reflection_prompt,
    query_writer_prompt,
    reflection_prompt,
    summarizer_prompt,
    tool_calling_query_prompt,
    tool_calling_reflection_prompt,
)
from local_deep_researcher.settings import refresh_settings
from local_deep_researcher.utils import (
    MAX_TOKENS_PER_SOURCE,
    SearchAPI,
    deduplicate_and_format_sources,
    exa_search,
    format_sources,
    get_llm,
    google_search,
    searxng_search,
    strip_thinking_tokens,
    tavily_search,
)

settings = refresh_settings()
T = TypeVar("T", bound=BaseModel)



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
        content: str = result.content  # type: ignore

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


def web_research(state: SummaryState, config: RunnableConfig) -> dict[str, Any]:
    """
    LangGraph node that performs web research using the generated search query.

    Executes a web search using the configured search API (tavily, perplexity,
    duckduckgo, or searxng) and formats the results for further processing.

    Parameters
    ----------
    state : SummaryState
        The current summary state containing the search query and research loop count
    config : RunnableConfig
        Configuration for the runnable, including search API settings

    Returns
    -------
    dict[str, Any]
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
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


def summarize_sources(state: SummaryState, config: RunnableConfig) -> Any:
    """
    LangGraph node that summarizes web research results.

    Uses an LLM to create or update a running summary based on the newest web research
    results, integrating them with any existing summary.

    Parameters
    ----------
    state : SummaryState
        The current summary state containing the research topic, running summary,
        and web research results
    config : RunnableConfig
        Configuration for the runnable, including LLM provider settings

    Returns
    -------
    dict[str, Any]
        Dictionary with state update, including running_summary key containing the updated summary
    """
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

    configurable = Configuration.from_runnable_config(config)
    if configurable.llm_provider == "llmstudio":
        # llm: ChatOpenAI = ChatOpenAI(
        #     api_key="empty",
        #     base_url=settings.LMSTUDIO_URL,  # type: ignore
        #     model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
        #     temperature=0,
        # )
        llm: ChatOpenAI = get_llm(configurable)
    else:
        llm = ChatOpenAI(
            api_key="empty",  # type: ignore
            base_url=settings.OLLAMA_URL,  # type: ignore
            model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
            temperature=0,
        )

    result = llm.invoke([SystemMessage(content=summarizer_prompt), HumanMessage(content=human_msg_content)])
    existing_summary: str = result.content  # type: ignore
    if configurable.strip_thinking_tokens:
        existing_summary = strip_thinking_tokens(existing_summary)

    return {"existing_summary": existing_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig) -> dict[str, Any]:
    """
    LangGraph node that identifies knowledge gaps and generates follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    a new search query to address those gaps. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
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


def finalize_summary(state: SummaryState) -> dict[str, Any]:
    """
    Parameters
    ----------
    state : SummaryState
        The current summary state containing the research topic, running summary,
        and web research results.

    Returns
    -------
    dict[str, Any]
        Dictionary with state update, including existing_summary key containing the updated summary
    """
    seen_sources: set = set()
    unique_sources: list[str] = []

    for src in state.sources_gathered:
        for line in src.split("\n"):
            # Process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    all_sources: str = "\n".join(unique_sources)
    state.existing_summary = f"## Summary:\n{state.existing_summary}\n\n### Sources: \n{all_sources}"

    return {"existing_summary": state.existing_summary}


def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """
    Route the research process based on the current research loop count.

    Parameters
    ----------
    state : SummaryState
        The current summary state containing the research loop count.
    config : RunnableConfig
        Configuration for the research process, including maximum allowed research loops.

    Returns
    -------
    Literal["finalize_summary", "web_research"]
        Returns "web_research" if the research loop count is less than or equal to the maximum allowed loops,
        otherwise returns "finalize_summary".
    """
    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    return "finalize_summary"


builder = StateGraph(
    state_schema=SummaryState,
    input_schema=SummaryStateInput,
    output_schema=SummaryStateOutput,
    config_schema=Configuration,
)

# Nodes
builder.add_node(generate_query, "generate_query")  # type: ignore
builder.add_node(web_research, "web_research")  # type: ignore
builder.add_node(summarize_sources, "summarize_sources")  # type: ignore
builder.add_node(reflect_on_summary, "reflect_on_summary")  # type: ignore
builder.add_node(finalize_summary, "finalize_summary")  # type: ignore

# Edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

# Build
research_graph = builder.compile()
