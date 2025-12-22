from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .services import get_container
from .routers import artifact_routes, cache_routes, config, documents, evaluation, health, monitoring, providers, query, traces

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services on startup
    # This triggers ServiceContainer.__init__, which builds the Orchestrator
    # and registers it in state.py, fixing the 503 error.
    print("Initializing Application Services...")
    container = get_container()
    print("Application Services Initialized.")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down Orchestrator...")
    if container.orchestrator:
        await container.orchestrator.stop()

app = FastAPI(title="JR AutoRAG API", version="0.1.0", lifespan=lifespan)

# CORS for local dev (web runs on 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(config.router, prefix="/config", tags=["config"])
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(evaluation.router)
app.include_router(monitoring.router)
app.include_router(providers.router)
app.include_router(traces.router, prefix="/api", tags=["traces"])
app.include_router(artifact_routes.router)
app.include_router(cache_routes.router)


@app.get("/")
def root():
    return {"name": "JR AutoRAG API", "status": "ok", "version": app.version}
