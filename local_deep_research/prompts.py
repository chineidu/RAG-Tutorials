from datetime import datetime


# Get current date in a readable format
def get_current_date() -> str:
    """Get the current date in a readable format."""
    return datetime.now().strftime("%B %d, %Y")


query_writer_prompt: str = """
<GOAL>Your goal is to generate an optimized web search query based on the user's query</GOAL>

<CONTEXT>
Current date: {current_date}
Please ensure your queries account for the most current available information using the latest data available.
</CONTEXT>

<TOPIC> {research_topic} </TOPIC>

<EXAMPLE>
{{
    "query": "retrieval augmented generation (rag) explained simply",
    "rationale": "understanding the fundamental concept of retrieval augmented generation (rag)"
}}
</EXAMPLE>
"""

json_mode_query_prompt: str = """

<GOAL>
    Format your response as a JSON object with EXACTLY the following keys:
    - "query": The actual search query
    - "rationale": Brief explanation why this query is relevant
</GOAL>

<REQUIREMENTS>
    - Output MUST have the following keys:
      - "query"
      - "rationale"
</REQUIREMENTS>

<OUTPUT> Please provide your response in the specified JSON format: </OUTPUT>
"""

tool_calling_query_prompt: str = """

<GOAL>
    Use the `Query` tool with these required fields:
    - query: search terms
    - rationale: why this query helps
</GOAL>

<REQUIREMENTS>
    - Call the `Query` tool with the REQUIRED arguments
    - Output MUST have the following keys:
      - "query"
      - "rationale"
</REQUIREMENTS>

<OUTPUT>
    Please provide your response in the specified JSON format:
</OUTPUT>
"""

summarizer_prompt: str = """
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
    When creating a NEW summary:
    - Highlight the most relevant information related to the user topic from the search results
    - Ensure a coherent flow of information

    When EXTENDING an existing summary:         
    - Read the existing summary and new search results carefully.
    - Compare the new information with the existing summary.     
    - For each piece of new information:                         
        a. If it's related to existing points, integrate it into the relevant paragraph.                               
        b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
        c. If it's not relevant to the user topic, skip it.        
    - Ensure all additions are relevant to the user's topic.     
    - Verify that your final output differs from the input summary.
</REQUIREMENTS>

<FORMATTING>
    - Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.
</FORMATTING>

<TASK>
    Think carefully about the provided Context first. Then generate a summary of the context to address the User Input.
</TASK>
"""

reflection_prompt: str = """
<ROLE>
    You are an expert research assistant analyzing a summary about {research_topic}.
    </ROLE>

    <GOAL>
    - Identify knowledge gaps or areas that need deeper exploration
    - Generate a follow-up question that would help expand your understanding
    - Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
    Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>
"""

json_mode_reflection_prompt: str = """
<FORMAT>
    Format your response as a JSON object with **EXACTLY** the following keys:
    - knowledge_gap: Describe what information is missing or needs clarification
    - follow_up_query: Write a specific question to address this gap
</FORMAT>

<TASK>
    - Reflect carefully on the summary to identify knowledge gaps and produce a follow-up query. Then, produce 
    your output following this JSON format:
    {{
        "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
        "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
    }}
</TASK>

Provide your analysis in JSON format:
"""

tool_calling_reflection_prompt: str = """
<INSTRUCTIONS>
    Call the `FollowUpQuery` tool to format your response with **EXACTLY** the following keys:
    - follow_up_query: Write a specific question to address this gap
    - knowledge_gap: Describe what information is missing or needs clarification
</INSTRUCTIONS>

<TASK>
    Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query.
</TASK>

<REQUIREMENTS>
    - Call the `FollowUpQuery` tool with the REQUIRED arguments
    - Output must match have the following keys:
      - "follow_up_query"
      - "knowledge_gap"
</REQUIREMENTS>

<OUTPUT>
    Please provide your response in the specified JSON format:
</OUTPUT>
"""
