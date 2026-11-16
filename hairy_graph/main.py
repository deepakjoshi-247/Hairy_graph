import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from hairy_graph import config
from hairy_graph.database import (
    get_connection,
    get_graph_stats,
    init_db,
)
from hairy_graph.graph_engine import (
    archive_last_memory,
    retrieve_context,
    save_memory,
)

logger = logging.getLogger("hairy_graph")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initializes database with tables and indexes
    init_db()
    yield


app = FastAPI(title="HairyGraph", lifespan=lifespan)

# CORS: explicitly allow all origins, do NOT allow credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security: Optional API Key verification
def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    expected_key = os.environ.get(config.API_KEY_ENV_VAR)
    if expected_key:
        if not x_api_key or not secrets.compare_digest(x_api_key, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
            )


# Request & Response Models
class GetContextRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=config.MAX_TEXT_LENGTH)
    max_tokens: int = Field(
        default=config.DEFAULT_MAX_TOKENS,
        ge=config.MIN_MAX_TOKENS,
        le=config.MAX_MAX_TOKENS,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class GetContextResponse(BaseModel):
    mode: str
    context: str
    tokens_used: int
    debug_trace: List[Dict[str, Any]]


class SaveMemoryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=config.MAX_TEXT_LENGTH)
    response: str = Field(default="", max_length=config.MAX_TEXT_LENGTH)
    pin_as_core: bool = False

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v


class SaveMemoryResponse(BaseModel):
    bubble_id: str
    cluster_id: str
    node_type: str
    archived_node_ids: List[str]
    promoted_node_ids: List[str]


class ArchiveLastResponse(BaseModel):
    archived_bubble_id: Optional[str]
    message: str


class GraphStatsResponse(BaseModel):
    total_clusters: int
    active_core: int
    active_state: int
    archived_bubbles: int


class HealthResponse(BaseModel):
    status: str


# Clean error response handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )
    logger.error("Unhandled error: %s", type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint. Remains publicly accessible even if API key auth is enabled."""
    return {"status": "ok"}


@app.post("/get_context", response_model=GetContextResponse, dependencies=[Depends(verify_api_key)])
def get_context_endpoint(payload: GetContextRequest):
    conn = get_connection()
    try:
        return retrieve_context(conn, query=payload.query, max_tokens=payload.max_tokens)
    finally:
        conn.close()


@app.post("/save_memory", response_model=SaveMemoryResponse, dependencies=[Depends(verify_api_key)])
def save_memory_endpoint(payload: SaveMemoryRequest):
    conn = get_connection()
    try:
        return save_memory(
            conn,
            prompt=payload.prompt,
            response=payload.response,
            pin_as_core=payload.pin_as_core,
        )
    finally:
        conn.close()


@app.post("/archive_last", response_model=ArchiveLastResponse, dependencies=[Depends(verify_api_key)])
def archive_last_endpoint():
    conn = get_connection()
    try:
        return archive_last_memory(conn)
    finally:
        conn.close()


@app.get("/graph_stats", response_model=GraphStatsResponse, dependencies=[Depends(verify_api_key)])
def graph_stats_endpoint():
    conn = get_connection()
    try:
        return get_graph_stats(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hairy_graph.main:app", host="127.0.0.1", port=8000, reload=False)
