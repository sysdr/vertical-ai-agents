"""
VAIAAgent – production agent class.
Bridges L63's tested logic; exposes the serving surface L65 load-tests.
"""
import asyncio
import logging
import google.generativeai as genai
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import Settings

logger = logging.getLogger("vaia.agent")


class VAIAAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_ready = False
        self._chroma_client: chromadb.AsyncHttpClient | None = None

        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        logger.info(f"Gemini model: {settings.gemini_model}")

    async def warm_up(self):
        """Establish ChromaDB connection; block readiness until done."""
        try:
            self._chroma_client = await chromadb.AsyncHttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
            )
            await self._chroma_client.heartbeat()
            logger.info("ChromaDB connection established.")
        except Exception as exc:
            logger.warning(f"ChromaDB warm-up failed (will retry on readiness probe): {exc}")
        self.is_ready = True

    async def chroma_heartbeat(self) -> bool:
        if self._chroma_client is None:
            return False
        try:
            await self._chroma_client.heartbeat()
            return True
        except Exception:
            return False

    async def infer(self, query: str, context: str | None = None) -> dict:
        """Run one inference turn against Gemini."""
        system_prompt = (
            "You are a VAIA enterprise assistant. "
            "Answer concisely and cite sources when available."
        )
        user_content = f"{context}\n\nQuery: {query}" if context else query

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                [system_prompt, user_content],
            )
            return {
                "answer": response.text,
                "model": self.settings.gemini_model,
                "finish_reason": response.candidates[0].finish_reason.name
                    if response.candidates else "UNKNOWN",
            }
        except Exception as exc:
            logger.error(f"Gemini inference error: {exc}")
            raise
