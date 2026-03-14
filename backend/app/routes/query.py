from fastapi import APIRouter, HTTPException

from app.schemas import QueryRequest, QueryResponse
from app.services.orchestrator import QueryOrchestrator
from app.utils.logger import logger

router = APIRouter()
orchestrator = QueryOrchestrator()


@router.post("/query", response_model=QueryResponse, tags=["Core"])
async def process_query(request: QueryRequest):
    """
    Main endpoint to process a natural language query.
    Orchestrates Gemini agents and MCP tools to return a grounded answer.
    """
    try:
        response = await orchestrator.process_query(request)
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the query.",
        )
