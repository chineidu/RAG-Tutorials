import sys
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from local_deep_researcher.api import get_agent, lifespan

router = APIRouter()


class UserInput(BaseModel):
    topic: str = Field(..., title="Research Topic", description="The topic to research")
    agent_id: str = Field("deep-researcher", title="Agent ID", description="The ID of the agent to use")
    user_id: str = Field(default="", title="User ID", description="A unique identifier for the user")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "topic": "Climate Change Impacts",
                    "agent_id": "deep-researcher",
                    "user_id": "",  # UUID will be auto-generated if not provided
                }
            ]
        }
    }


def handle_user_input(user_input: UserInput) -> dict[str, Any]:
    thread_id: str = str(user_input.user_id) if user_input.user_id else str(uuid4())
    return {"thread_id": thread_id}


@router.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/invoke", tags=["Invocation"])
async def ainvoke(user_input: UserInput) -> dict[str, str | Any]:
    try:
        user_input_dict: dict[str, Any] = handle_user_input(user_input)
        user_id: str = user_input_dict["thread_id"]
        config: dict[str, Any] = {"configurable": {"thread_id": user_id}}
        agent: CompiledStateGraph | None = get_agent(user_input.agent_id)

        responses: list[dict[str, Any]] = [
            event
            async for event in agent.astream(  # type: ignore
                {"research_topic": user_input.topic},
                config=config,
            )
        ]
        final_response = responses[-1].get("finalize_summary").get("existing_summary")
        return {"user_id": user_id, "response": final_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def create_application() -> FastAPI:
    """Create and configure a FastAPI application instance.

    This function initializes a FastAPI application with custom configuration settings,
    adds CORS middleware, and includes API route handlers.

    Returns
    -------
    FastAPI
        A configured FastAPI application instance.
    """
    app = FastAPI(
        title="Local Deep Researcher API",
        description="API for the Local Deep Researcher",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router, prefix="/api/v1")
    # app.include_router(health.router, prefix=app_config.api.prefix)
    # app.include_router(task_status.router, prefix=app_config.api.prefix)

    return app


app: FastAPI = create_application()

if __name__ == "__main__":
    try:
        uvicorn.run(
            "local_deep_researcher.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )
    except (Exception, KeyboardInterrupt) as e:
        print(f"Error creating application: {e}")
        print("Exiting gracefully...")
        sys.exit(1)
