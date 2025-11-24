from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import config, documents, evaluation, health, monitoring, providers, query

app = FastAPI(title="JR AutoRAG API", version="0.1.0")

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


@app.get("/")
def root():
    return {"name": "JR AutoRAG API", "status": "ok", "version": app.version}
