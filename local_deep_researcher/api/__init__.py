import logging
import pathlib
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, _AsyncGeneratorContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from local_deep_researcher.graph import research_graph  # type: ignore

logger = logging.getLogger(__name__)


class AsyncInMemoryStore:
    """Wrapper for InMemoryStore that provides an async context manager interface."""

    def __init__(self) -> None:
        self.store = InMemoryStore()

    async def __aenter__(self) -> InMemoryStore:
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # No cleanup needed for InMemoryStore
        pass

    async def setup(self) -> None:
        """Set up the in-memory store."""
        # No-op method for compatibility with PostgresStore
        pass


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """Initialize and return a SQLite saver instance."""
    # Use absolute path to avoid CWD issues
    db_path = pathlib.Path(__file__).parent.parent.parent / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    conn_string: str = str(db_path)

    return AsyncSqliteSaver.from_conn_string(conn_string)


@asynccontextmanager
async def get_sqlite_store() -> AsyncGenerator[InMemoryStore, None]:
    """Initialize and return a store instance for long-term memory.

    Note: SQLite-specific store isn't available in LangGraph,
    so we use InMemoryStore wrapped in an async context manager for compatibility.
    """
    store_manager = AsyncInMemoryStore()
    yield await store_manager.__aenter__()


def initialize_database() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """
    Initialize the appropriate database checkpointer based on configuration.
    Returns an initialized AsyncCheckpointer instance.
    """
    # if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
    #     return get_postgres_saver()
    # if settings.DATABASE_TYPE == DatabaseType.MONGO:
    #     return get_mongo_saver()
    return get_sqlite_saver()


def initialize_store() -> _AsyncGeneratorContextManager[InMemoryStore, None]:
    """
    Initialize the appropriate store based on configuration.
    Returns an async context manager for the initialized store.
    """
    # if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
    #     return get_postgres_store()
    # # TODO: Add Mongo store - https://pypi.org/project/langgraph-store-mongodb/
    # Default to SQLite
    return get_sqlite_store()


@dataclass
class Agent:
    name: str
    description: str
    agent: CompiledStateGraph


agents: list[dict[str, Any]] = [
    {
        "id": "deep-researcher",
        "agent": Agent(
            name="Deep Researcher",
            description="A research assistant that performs deep web research on a given topic, "
            "iteratively refining its understanding and summarizing findings.",
            agent=research_graph,
        ),
    }
]


def get_agent(id: str) -> CompiledStateGraph | None:
    """
    Retrieve an agent by ID.
    """
    for a in agents:
        if a["id"] == id:
            return a["agent"].agent

    print(f"Agent '{id}' not found.")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """
    Configurable lifespan that initializes the appropriate database checkpointer and store
    based on settings.
    """
    try:
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # Set up both components
            if hasattr(saver, "setup"):
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):
                await store.setup() # type: ignore

            # Configure agents with both memory components
            # agents = get_all_agent_info()
            for a in agents:
                agent: CompiledStateGraph | None = get_agent(a["id"])
                if agent is None:
                    continue
                # Set checkpointer for thread-scoped memory (conversation history)
                agent.checkpointer = saver  # type: ignore
                # Set store for long-term memory (cross-conversation knowledge)
                agent.store = store  # type: ignore
            yield
            
    except Exception as e:
        logger.error(f"Error during database/store initialization: {e}")
        raise
