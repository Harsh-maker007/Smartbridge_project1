"""
src/gemini/client.py
======================
Google Gemini API client wrapper with retry logic and error handling.
"""

import time
import google.generativeai as genai
from typing import Optional

from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """
    Wrapper around google-generativeai SDK.
    Provides retry logic, rate-limit handling, and clean error messages.
    """

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key    = api_key    or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL
        self._model: Optional[genai.GenerativeModel] = None

        if not self.api_key:
            raise ValueError(
                "Gemini API key not set. Add GEMINI_API_KEY to your .env file."
            )

    def _get_model(self) -> genai.GenerativeModel:
        """Lazily initialize the Gemini model."""
        if self._model is None:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    top_p=0.95,
                ),
            )
            logger.info(f"Gemini model initialized: {self.model_name}")
        return self._model

    def generate(self, prompt: str, max_retries: int = 3, retry_delay: float = 2.0) -> str:
        """
        Generate text from a prompt with automatic retry on transient errors.

        Args:
            prompt:      The full prompt string.
            max_retries: Number of retry attempts on failure.
            retry_delay: Seconds to wait between retries.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If all retries fail.
        """
        model = self._get_model()

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Gemini call attempt {attempt}/{max_retries}")
                response = model.generate_content(prompt)

                if response.text:
                    return response.text.strip()
                else:
                    logger.warning("Gemini returned empty response.")
                    return self._fallback_response()

            except Exception as e:
                error_str = str(e).lower()

                if "quota" in error_str or "rate" in error_str:
                    logger.warning(f"Rate limit hit. Waiting {retry_delay * attempt}s...")
                    time.sleep(retry_delay * attempt)
                elif "api_key" in error_str or "invalid" in error_str:
                    raise RuntimeError(f"Invalid Gemini API key: {e}")
                else:
                    logger.error(f"Gemini error (attempt {attempt}): {e}")
                    if attempt == max_retries:
                        logger.error("All Gemini retries exhausted. Using fallback.")
                        return self._fallback_response()
                    time.sleep(retry_delay)

        return self._fallback_response()

    @staticmethod
    def _fallback_response() -> str:
        """Return a generic response when Gemini API is unavailable."""
        return (
            "**Guidance Temporarily Unavailable**\n\n"
            "We're having trouble connecting to the AI guidance service right now. "
            "Here are some general tips:\n\n"
            "- 🧘 Take a short break (5-10 minutes) to reset your focus.\n"
            "- 📖 Review your notes or textbook for context.\n"
            "- 🔍 Search for the concept on Khan Academy or YouTube.\n"
            "- 💬 Reach out to a classmate or teacher for help.\n\n"
            "Try again in a moment!"
        )
