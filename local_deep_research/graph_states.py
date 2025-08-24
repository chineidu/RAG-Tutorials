import operator
from dataclasses import dataclass, field
from typing import Annotated


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)  # type: ignore
    search_query: str = field(default=None)  # type: ignore
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)  # Research loop count
    existing_summary: str = field(default=None)  # Final report # type: ignore


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)  # type: ignore


@dataclass(kw_only=True)
class SummaryStateOutput:
    existing_summary: str = field(default=None)  # type: ignore
