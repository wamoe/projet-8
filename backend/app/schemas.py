from pydantic import BaseModel, Field


# --- API Request / Response Schemas ---

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's natural language query")


class Source(BaseModel):
    title: str
    url: str
    description: str
    reason_for_selection: str
    confidence_score: float


class QueryResponse(BaseModel):
    user_query: str
    selected_sources: list[Source]
    answer: str
    limitations: list[str]
    trace: list[str]


# --- Internal Agent Schemas (Used for Gemini Structured Output) ---

class OrchestratorPlan(BaseModel):
    search_queries: list[str] = Field(
        description="List of search queries to send to data.gouv.fr"
    )
    reasoning: str = Field(description="Why these queries were chosen")


class ScoutSelection(BaseModel):
    selected_sources: list[Source] = Field(
        description="The filtered and scored list of relevant sources"
    )


class SynthesizedAnswer(BaseModel):
    answer: str = Field(description="The final synthesized answer")
    limitations: list[str] = Field(
        description="List of limitations or missing data"
    )
