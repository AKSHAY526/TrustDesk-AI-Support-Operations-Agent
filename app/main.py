"""FastAPI app. Endpoints are plain ``def`` functions, so FastAPI runs them in a
threadpool and a slow model call doesn't hold up other requests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import DB_PATH, settings
from .db import init_db
from .embeddings import get_embedder
from .llm import get_llm
from .routers import actions, insights, kb, tickets
from .tools import ToolError

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

DESCRIPTION = """
An AI support agent that reads a customer ticket, finds the right company policy,
and writes a reply showing which policy it used. It may suggest an action such as
sending a replacement, but a manager must approve anything involving money.

**Authentication** - send `Authorization: Bearer <token>`. The demo ships one
token per role (see `.env.example`): agent, manager and admin.

**Design note** - the model proposes; the engine decides. Approval, amount caps,
category permissions and idempotency are enforced in code and in the database, so
a misled model cannot exceed them.
"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create the schema and load the search model before the first request."""
    conn = init_db()
    conn.close()
    get_embedder()  # loading it on first use made the console's first page load crawl
    yield


app = FastAPI(
    title="TrustDesk - AI Support Operations Agent",
    version="0.1.0",
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.include_router(tickets.router)
app.include_router(kb.router)
app.include_router(actions.router)
app.include_router(insights.router)

if STATIC_DIR.exists():
    app.mount("/console", StaticFiles(directory=str(STATIC_DIR), html=True), name="console")


@app.exception_handler(ToolError)
def handle_tool_error(request, exc: ToolError) -> JSONResponse:  # noqa: ARG001
    """Surface catalogue rejections with their stable error code."""
    return JSONResponse(status_code=exc.http_status, content={"detail": exc.to_detail()})


@app.get("/health", tags=["meta"], summary="Liveness and configuration summary")
def health() -> dict:
    """Reports which model and retrieval settings are actually in use."""
    return {
        "status": "ok",
        "database": DB_PATH.name,
        "model": getattr(get_llm(), "name", "unknown"),
        "using_real_model": settings.use_real_model,
        "embedder": get_embedder().name,
        "retrieval_min_score": settings.retrieval_min_score,
    }


@app.get("/", include_in_schema=False, response_model=None)
def root():
    """Serve the console if it is present, otherwise point at the API docs."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"service": "TrustDesk", "docs": "/docs", "console": "/console"})
