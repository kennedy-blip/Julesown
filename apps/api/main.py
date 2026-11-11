import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import webhooks, tasks, health

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

# Register health router
app.include_router(health.router)

# Register webhooks router
app.include_router(webhooks.router)

# Register tasks router for both /api/v1/tasks and /api/tasks to guarantee zero 404s
app.include_router(tasks.router, prefix="/api/v1/tasks")
app.include_router(tasks.router, prefix="/api/tasks")

@app.get("/api/health")
async def legacy_health_check():
    """Direct /api/health endpoint mirroring /api/v1/health."""
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
