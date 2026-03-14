from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    """Returns the health status of the API."""
    return {"status": "ok", "service": "DataGouv Alive API"}
