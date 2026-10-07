"""
Groq LLM Reasoning Service for VidSense AI.

Takes user questions, current playback timestamps, and associated events
retrieved from MongoDB, then asks Groq (Llama 3.3 70B) to reason over the
temporal events and provide timestamped answers and evidence references.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


class GroqReasoningService:
    """Temporal reasoning engine powered by Groq LLM."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.client: Optional[Groq] = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)

    def answer_query(
        self,
        query: str,
        events: List[Dict[str, Any]],
        current_time: Optional[float] = None,
        time_range_start: Optional[float] = None,
        time_range_end: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ask Groq to answer the user query based on MongoDB events and video playback context.
        """
        if not events:
            return {
                "answer": "No events are available in the database for this video segment.",
                "start_time": current_time or 0.0,
                "end_time": (current_time or 0.0) + 5.0,
                "confidence": 0.0,
                "evidence_frames": [],
                "key_objects": [],
            }

        if not self.client:
            logger.warning("GROQ_API_KEY is not configured in .env.")
            first_event = events[0]
            return {
                "answer": f"GROQ_API_KEY is not set in .env. Timeline event at {first_event.get('start_time', 0.0):.1f}s: {first_event.get('description', '')[:180]}",
                "start_time": first_event.get("start_time", 0.0),
                "end_time": first_event.get("end_time", 5.0),
                "confidence": 0.5,
                "evidence_frames": [],
                "key_objects": [f"Track #{t}" for t in first_event.get("track_ids", [])[:3]],
            }

        # Build events summary for the prompt
        events_context_lines = []
        for e in events:
            st = e.get("start_time", 0.0)
            et = e.get("end_time", 0.0)
            desc = e.get("description", "")
            t_ids = e.get("track_ids", [])
            etype = e.get("event_type", "activity")
            events_context_lines.append(
                f"- [{st:05.1f}s - {et:05.1f}s] ({etype}, Tracks: {t_ids}): {desc}"
            )
        events_context = "\n".join(events_context_lines)

        user_context_str = ""
        if current_time is not None:
            user_context_str += f"\nUser currently paused/stopped at timestamp: {current_time:.2f}s."
        if time_range_start is not None and time_range_end is not None:
            user_context_str += (
                f"\nUser focused time window: {time_range_start:.2f}s to {time_range_end:.2f}s."
            )

        system_prompt = (
            "You are VidSense AI, an intelligent video understanding and temporal reasoning assistant.\n"
            "You are given structured chronological video events extracted by computer vision (YOLO + BoT-SORT) and a vision-language model.\n"
            "Your task is to answer the user's question accurately and concisely based strictly on the provided events.\n\n"
            "STRICT RULES:\n"
            "1. Base your answer only on what is explicitly stated in the events.\n"
            "2. Identify the most relevant start and end timestamps (in seconds) corresponding to the answer.\n"
            "3. Mention specific actors or tracked objects (e.g. 'Person #1', 'Skateboard #2') when relevant.\n"
            "4. Return your response in VALID JSON format with exactly the following keys:\n"
            "   {\n"
            '     "answer": "Concise 1-3 sentence direct answer to the user query.",\n'
            '     "start_time": <float start timestamp in seconds>,\n'
            '     "end_time": <float end timestamp in seconds>,\n'
            '     "confidence": <float between 0.0 and 1.0>,\n'
            '     "key_objects": ["Person #1", "Skateboard #2", ...]\n'
            "   }\n"
            "Do not include any conversational filler outside the JSON."
        )

        user_prompt = (
            f"VIDEO EVENTS TIMELINE:\n{events_context}\n"
            f"{user_context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            "Please return the JSON object answering this question:"
        )

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=600,
                response_format={"type": "json_object"},
            )

            raw_response = chat_completion.choices[0].message.content or "{}"
            result = json.loads(raw_response)

            # Ensure expected fields
            start_t = float(result.get("start_time", events[0].get("start_time", 0.0)))
            end_t = float(result.get("end_time", events[0].get("end_time", 5.0)))

            return {
                "answer": result.get("answer", "Activity detected in this video segment."),
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "confidence": float(result.get("confidence", 0.9)),
                "key_objects": result.get("key_objects", []),
            }

        except Exception as e:
            logger.error("Groq API error: %s", e)
            # Fallback
            first_event = events[0]
            return {
                "answer": f"At {first_event.get('start_time', 0.0):.1f}s: {first_event.get('description', '')[:180]}",
                "start_time": first_event.get("start_time", 0.0),
                "end_time": first_event.get("end_time", 5.0),
                "confidence": 0.8,
                "key_objects": [f"Track #{t}" for t in first_event.get("track_ids", [])[:3]],
            }


_groq_instance: Optional[GroqReasoningService] = None


def get_groq_service() -> GroqReasoningService:
    global _groq_instance
    if _groq_instance is None:
        _groq_instance = GroqReasoningService()
    return _groq_instance
