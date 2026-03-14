from typing import Any

import httpx

from app.config import settings
from app.utils.logger import logger


class MCPService:
    """
    Abstraction layer for data.gouv.fr MCP tool interactions.
    Replace _get_mock_data with real MCP calls when the server is available.
    """

    def __init__(self):
        self.use_mock = settings.use_mock_mcp
        self.mcp_url = settings.mcp_server_url

    async def search_data_gouv(self, queries: list[str]) -> list[dict[str, Any]]:
        """
        Searches data.gouv.fr via the MCP tool layer.

        In mock mode, returns static sample data.
        In real mode, sends queries to the configured MCP server endpoint.
        """
        if self.use_mock:
            logger.info(f"Mocking MCP search for queries: {queries}")
            return self._get_mock_data()

        # TODO: Replace with real MCP integration when MCP server is running.
        # Example:
        # async with httpx.AsyncClient() as client:
        #     try:
        #         response = await client.post(
        #             f"{self.mcp_url}/tools/datagouv_search",
        #             json={"queries": queries},
        #             timeout=10.0,
        #         )
        #         response.raise_for_status()
        #         return response.json().get("results", [])
        #     except Exception as e:
        #         logger.error(f"MCP Error: {e}")
        #         return []

        return []

    def _get_mock_data(self) -> list[dict[str, Any]]:
        """Returns static mock datasets for demo mode."""
        return [
            {
                "id": "5448d3e0c751df01f85d0572",
                "title": (
                    "Fichier consolidé des Bornes de Recharge pour "
                    "Véhicules Électriques (IRVE)"
                ),
                "description": (
                    "Base de données consolidée des infrastructures de recharge "
                    "publique pour véhicules électriques."
                ),
                "url": (
                    "https://www.data.gouv.fr/fr/datasets/"
                    "fichier-consolide-des-bornes-de-recharge-pour-vehicules-electriques-irve/"
                ),
                "organization": "Ministère de la Transition écologique",
            },
            {
                "id": "60a3927b271101918a1a3e81",
                "title": "Immatriculations des véhicules électriques",
                "description": (
                    "Données sur les immatriculations de véhicules électriques "
                    "et hybrides par région."
                ),
                "url": (
                    "https://www.data.gouv.fr/fr/datasets/"
                    "immatriculations-vehicules-electriques/"
                ),
                "organization": "Ministère de l'Intérieur",
            },
        ]
