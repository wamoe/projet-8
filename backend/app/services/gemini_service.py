from typing import Type, TypeVar

from pydantic import BaseModel

from app.config import settings
from app.utils.logger import logger

T = TypeVar("T", bound=BaseModel)


class GeminiService:
    """Wraps Google GenAI SDK calls with mock fallback support."""

    def __init__(self):
        self.use_mock = settings.use_mock_gemini
        if not self.use_mock:
            if not settings.gemini_api_key:
                logger.warning(
                    "GEMINI_API_KEY is missing. Falling back to mock mode."
                )
                self.use_mock = True
            else:
                # TODO: Replace with real Gemini client once API key is available
                from google import genai  # noqa: PLC0415  (import inside __init__ is intentional)

                self.client = genai.Client(api_key=settings.gemini_api_key)
                self.model_name = "gemini-2.5-flash"

    def generate_structured(
        self, prompt: str, response_model: Type[T], mock_response: T
    ) -> T:
        """
        Calls Gemini and forces the output to match the provided Pydantic model.
        Falls back to mock_response if in mock mode or if an error occurs.
        """
        if self.use_mock:
            logger.info(f"Mocking Gemini response for {response_model.__name__}")
            return mock_response

        try:
            from google.genai import types  # noqa: PLC0415  (import inside try block is intentional)

            logger.info(f"Calling Gemini for {response_model.__name__}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_model,
                    temperature=0.2,
                ),
            )
            # Parse the JSON string returned by Gemini into the Pydantic model
            return response_model.model_validate_json(response.text)

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}. Falling back to mock.")
            return mock_response
