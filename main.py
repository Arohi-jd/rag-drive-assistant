"""
Main entry point for the RAG Drive Assistant FastAPI application.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Import routes
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    try:
        print("🚀 Starting RAG Drive Assistant...")
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        display_host = "localhost" if host in {"0.0.0.0", "::"} else host
        server_url = f"http://{display_host}:{port}"
        docs_url = f"{server_url}/docs"

        print("✨ Welcome to RAG Drive Assistant")
        print(f"🌍 Server URL: {server_url}")
        print(f"📘 Docs URL: {docs_url}")
        print("📚 Documents indexed: 0 (lazy load enabled)")
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
    
    yield
    
    try:
        print("🛑 Shutting down RAG Drive Assistant...")
        # Cleanup if needed
        print("✅ Shutdown completed")
    except Exception as e:
        print(f"❌ Error during shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title="RAG Drive Assistant",
    description="Retrieval-Augmented Generation system with Google Drive integration",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api", tags=["RAG"])


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return empty favicon response to avoid noisy 404 logs."""
    try:
        return Response(status_code=204)
    except Exception as e:
        print(f"❌ Favicon endpoint failed: {str(e)}")
        return Response(status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "message": "RAG Drive Assistant is running"
        }
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    
    try:
        print("🌍 Starting server on 0.0.0.0")
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    except Exception as e:
        print(f"❌ Failed to start server: {str(e)}")
