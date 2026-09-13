import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from app.core.config import settings
from app.core.logging import RequestIdMiddleware, logger
from db.database import init_db

# Import routers
from app.api.routes_health import router as health_router
from app.api.routes_query import router as query_router
from app.api.routes_documents import router as documents_router
from app.api.routes_ingestion import router as ingestion_router
from app.api.routes_evaluation import router as evaluation_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database schemas & vector tables
    init_db()
    logger.info("RAG Document Q&A API initialized successfully.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Document Q&A API",
        description=(
            "Production-grade Enterprise RAG system featuring multi-strategy chunking, "
            "hybrid retrieval (pgvector + BM25), Reciprocal Rank Fusion, Cross-Encoder re-ranking, "
            "query rewriting, claim-level citation verification, and empirical evaluation."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    # 1. Request ID & Structured Logging Middleware
    app.add_middleware(RequestIdMiddleware)

    # 2. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Register API Routers (mounted both at root and /api for full client compatibility)
    routers = [
        health_router,
        query_router,
        documents_router,
        ingestion_router,
        evaluation_router,
    ]

    for r in routers:
        app.include_router(r)
        app.include_router(r, prefix="/api")

    # 4. SPA Frontend Mount (Zero-Docker static serving fallback)
    frontend_dist = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "rag-frontend", "dist")
    )
    if os.path.exists(frontend_dist):
        assets_dir = os.path.join(frontend_dist, "assets")
        if os.path.exists(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="static_assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            if (
                full_path.startswith("api/")
                or full_path == "api"
                or full_path.startswith("docs")
                or full_path.startswith("openapi.json")
            ):
                raise HTTPException(status_code=404, detail="API endpoint not found.")

            file_path = os.path.join(frontend_dist, full_path)
            if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
                return FileResponse(file_path)

            index_file = os.path.join(frontend_dist, "index.html")
            if os.path.exists(index_file):
                return FileResponse(index_file)

            raise HTTPException(status_code=404, detail="SPA index.html not found.")

    return app


app = create_app()
