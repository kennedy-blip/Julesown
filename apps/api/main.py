import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import webhooks, tasks

# Initialize FastAPI app configured for Vercel Serverless Functions
app = FastAPI(
    title="Jules Agent Platform API",
    description="Autonomous coding agent serverless API with GitHub webhooks, E2B microVMs, and LLM tool execution.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Configure CORS for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(webhooks.router)
app.include_router(tasks.router)

@app.get("/api/health")
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint for Vercel deployment verification."""
    return {
        "status": "ok",
        "service": "jules-clone-api",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@app.get("/")
async def root():
    return {
        "message": "Jules Autonomous Agent API is live.",
        "docs": "/api/docs"
    }
