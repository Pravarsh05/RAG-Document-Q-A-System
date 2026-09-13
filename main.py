"""
Main entrypoint for the RAG Document Q&A System.
Re-exports the application instance from the modular app package.
"""
import sys
import os

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import app, create_app
from app.core.config import settings

__all__ = ["app", "create_app", "settings"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=(settings.APP_ENV == "development"),
    )
