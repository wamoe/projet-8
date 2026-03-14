from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import demo, health, query
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DataGouv Alive API is starting up...")
    yield
    logger.info("DataGouv Alive API is shutting down...")


app = FastAPI(
    title="DataGouv Alive API",
    description=(
        "Backend MVP for querying data.gouv.fr via MCP and Gemini Agents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(demo.router)
app.include_router(query.router)
