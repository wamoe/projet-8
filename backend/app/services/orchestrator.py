from pathlib import Path

from app.schemas import (
    OrchestratorPlan,
    QueryRequest,
    QueryResponse,
    ScoutSelection,
    Source,
    SynthesizedAnswer,
)
from app.services.gemini_service import GeminiService
from app.services.mcp_service import MCPService
from app.utils.logger import logger


class QueryOrchestrator:
    """
    Coordinates the three-agent pipeline:
      1. Orchestrator  – analyses the query and plans search terms.
      2. Dataset Scout – selects the most relevant data.gouv.fr sources.
      3. Answer Synthesizer – produces the final grounded answer.
    """

    def __init__(self):
        self.gemini = GeminiService()
        self.mcp = MCPService()
        self.prompts_dir = Path(__file__).parent.parent / "prompts"

    def _load_prompt(self, filename: str, **kwargs) -> str:
        """Load a prompt template from disk and format it with the given kwargs."""
        filepath = self.prompts_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            template = f.read()
        return template.format(**kwargs)

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Run the full agent pipeline for a user query."""
        trace: list[str] = []
        logger.info(f"Processing query: {request.query}")
        trace.append("Started query processing.")

        # --- Agent 1: Orchestrator ---
        trace.append("Agent 1 (Orchestrator) analysing query...")
        plan_prompt = self._load_prompt("orchestrator.txt", query=request.query)
        mock_plan = OrchestratorPlan(
            search_queries=[
                "bornes recharge vehicules electriques",
                "IRVE France",
            ],
            reasoning="Looking for electric vehicle charging station datasets.",
        )
        plan = self.gemini.generate_structured(
            plan_prompt, OrchestratorPlan, mock_plan
        )
        trace.append(f"Orchestrator decided to search for: {plan.search_queries}")

        # --- MCP Tool Execution ---
        trace.append("Calling MCP data.gouv.fr tools...")
        raw_results = await self.mcp.search_data_gouv(plan.search_queries)
        trace.append(f"MCP returned {len(raw_results)} raw datasets.")

        # --- Agent 2: Dataset Scout ---
        trace.append("Agent 2 (Dataset Scout) filtering results...")
        scout_prompt = self._load_prompt(
            "dataset_scout.txt",
            query=request.query,
            raw_results=str(raw_results),
        )
        mock_scout = ScoutSelection(
            selected_sources=[
                Source(
                    title="Fichier consolidé des Bornes de Recharge (IRVE)",
                    url=(
                        "https://www.data.gouv.fr/fr/datasets/"
                        "fichier-consolide-des-bornes-de-recharge-pour-vehicules-electriques-irve/"
                    ),
                    description=(
                        "Consolidated database of public EV charging stations."
                    ),
                    reason_for_selection=(
                        "Directly answers the query with official consolidated data."
                    ),
                    confidence_score=0.95,
                )
            ]
        )
        selection = self.gemini.generate_structured(
            scout_prompt, ScoutSelection, mock_scout
        )
        trace.append(
            f"Scout selected {len(selection.selected_sources)} relevant sources."
        )

        # --- Agent 3: Answer Synthesizer ---
        trace.append("Agent 3 (Answer Synthesizer) generating final response...")
        synth_prompt = self._load_prompt(
            "answer_synthesizer.txt",
            query=request.query,
            sources=selection.model_dump_json(),
        )
        mock_synth = SynthesizedAnswer(
            answer=(
                "Based on data.gouv.fr, the most relevant dataset is the "
                "'Fichier consolidé des Bornes de Recharge pour Véhicules "
                "Électriques (IRVE)'. It provides a consolidated view of "
                "public charging infrastructure across France."
            ),
            limitations=[
                "The dataset only covers public charging stations, "
                "not private residential ones."
            ],
        )
        synthesis = self.gemini.generate_structured(
            synth_prompt, SynthesizedAnswer, mock_synth
        )
        trace.append("Synthesis complete.")

        return QueryResponse(
            user_query=request.query,
            selected_sources=selection.selected_sources,
            answer=synthesis.answer,
            limitations=synthesis.limitations,
            trace=trace,
        )
