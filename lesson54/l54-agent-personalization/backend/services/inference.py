"""
PreferenceInferenceEngine — uses Gemini 2.0 Flash to derive behavioral
preference vectors from raw interaction history.
"""
from __future__ import annotations
import json
import logging
import numpy as np
from typing import Optional
import google.generativeai as genai

from ..models.profile import PreferenceVector
from ..db.store import UserProfileStore
from .router import AdaptiveAgentRouter

logger = logging.getLogger(__name__)

INFERENCE_PROMPT = """You are an AI behavioral analyst. Analyze the following conversation history between a user and an AI agent. Infer the user's communication style preferences.

CONVERSATION HISTORY:
{corpus}

Analyze patterns in:
- How long the user's messages are (verbosity)
- Formality of language
- How technical the topics they pursue are
- Whether they ask for examples
- How comfortable they are with uncertainty ("maybe", "approximately")
- Whether they prefer narrative explanations or bullet lists
- How broad vs narrow their topics are
- How urgent or patient they seem

Return ONLY valid JSON (no markdown, no preamble) with this exact structure:
{{
  "verbosity": <0.0-1.0>,
  "formality": <0.0-1.0>,
  "technical_depth": <0.0-1.0>,
  "example_density": <0.0-1.0>,
  "hedge_tolerance": <0.0-1.0>,
  "narrative_vs_list": <0.0-1.0>,
  "domain_breadth": <0.0-1.0>,
  "response_urgency": <0.0-1.0>,
  "reasoning": "<one sentence rationale>"
}}"""

DRIFT_THRESHOLD = 0.3  # L2 distance trigger


class PreferenceInferenceEngine:
    def __init__(self, store: UserProfileStore, api_key: str,
                 privacy_mode: bool = True):
        self.store = store
        self.privacy_mode = privacy_mode
        self.router = AdaptiveAgentRouter()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def infer_and_store(self, user_id: str) -> dict:
        """Full inference pipeline: history → vector → persona → store → drift check."""
        profile = await self.store.get_profile(user_id)
        if not profile:
            return {"error": "Profile not found", "user_id": user_id}

        if not profile.get("consent_behavioral"):
            return {"error": "User has not consented to behavioral tracking", "user_id": user_id}

        history = await self.store.get_recent_interactions(user_id, limit=20)
        if not history:
            return {"status": "insufficient_data", "message": "Need at least 1 interaction", "user_id": user_id}

        # Build corpus
        corpus_lines = []
        for h in reversed(history):  # chronological
            corpus_lines.append(f"USER: {h['user_msg'][:300]}")
            corpus_lines.append(f"AGENT: {h['agent_msg'][:300]}")
        corpus = "\n".join(corpus_lines)

        # Gemini inference
        prompt = INFERENCE_PROMPT.format(corpus=corpus)
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            # Strip possible markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw = json.loads(raw_text)
            reasoning = raw.pop("reasoning", "")
            new_vector = PreferenceVector.from_dict(raw)
        except Exception as e:
            logger.warning("Inference unavailable for %s (%s); using fallback vector", user_id, e)
            merged = {**PreferenceVector.default().__dict__, **profile.get("explicit_preferences", {})}
            new_vector = PreferenceVector.from_dict(merged)
            reasoning = "Fallback: Gemini unavailable or quota-limited; derived from explicit preferences."
            fallback_persona = self.router.select_persona(new_vector)

            await self.store.upsert_preference_vector(
                user_id,
                vars(new_vector.add_laplace_noise(epsilon=0.1) if self.privacy_mode else new_vector),
                fallback_persona.value,
            )
            return {
                "status": "fallback",
                "user_id": user_id,
                "persona": fallback_persona.value,
                "preference_vector": vars(new_vector),
                "reasoning": reasoning,
                "interactions_analyzed": len(history),
                "privacy_noise_applied": self.privacy_mode,
                "message": "Gemini quota exceeded/unavailable. Using fallback inference."
            }

        # Apply differential privacy for export safety
        export_vector = new_vector.add_laplace_noise(epsilon=0.1) if self.privacy_mode else new_vector

        # Drift detection (L54 assignment pattern)
        drift_result = None
        old_vec_data = profile.get("preference_vector")
        if old_vec_data:
            old_vector = PreferenceVector.from_dict(old_vec_data)
            l2 = new_vector.l2_distance(old_vector)
            if l2 > DRIFT_THRESHOLD:
                # Find largest changed dimension
                old_arr = np.array(old_vector.to_list())
                new_arr = np.array(new_vector.to_list())
                deltas = np.abs(new_arr - old_arr)
                dim_names = list(PreferenceVector.__dataclass_fields__.keys())
                max_idx = int(np.argmax(deltas))
                changed_dim = dim_names[max_idx]
                delta = float(deltas[max_idx])
                await self.store.log_preference_drift(user_id, l2, changed_dim, delta)
                drift_result = {
                    "drift_detected": True,
                    "l2_distance": round(l2, 4),
                    "changed_dimension": changed_dim,
                    "delta": round(delta, 4)
                }

        # Select persona
        persona = self.router.select_persona(new_vector)

        # Store
        await self.store.upsert_preference_vector(
            user_id, vars(export_vector), persona.value
        )

        result = {
            "status": "success",
            "user_id": user_id,
            "persona": persona.value,
            "preference_vector": vars(new_vector),
            "reasoning": reasoning,
            "interactions_analyzed": len(history),
            "privacy_noise_applied": self.privacy_mode
        }
        if drift_result:
            result["drift"] = drift_result

        logger.info(f"Inference complete for {user_id}: persona={persona.value}")
        return result
