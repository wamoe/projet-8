from fastapi import APIRouter

router = APIRouter()


@router.get("/demo/scenarios", tags=["Demo"])
async def get_scenarios():
    """Returns pre-configured demo scenarios for the hackathon UI."""
    return {
        "scenarios": [
            "What are the latest datasets about electric vehicle charging stations in France?",
            "Find data on water quality in Paris.",
            "Are there any datasets regarding public transport delays in Lyon?",
        ]
    }
