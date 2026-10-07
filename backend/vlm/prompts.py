"""
VLM prompt templates for Qwen2.5-VL video understanding.

All prompts live here so they can be iterated independently.
"""

from __future__ import annotations

from typing import Dict, List


def build_clip_analysis_prompt(
    start_time: float,
    end_time: float,
    track_summaries: List[Dict],
) -> str:
    """
    Build the system/user prompt for a temporal clip.

    Parameters
    ----------
    start_time : clip start in seconds.
    end_time : clip end in seconds.
    track_summaries : list of dicts with keys like
        ``{"track_id": 7, "class_name": "person", ...}``

    Returns
    -------
    The user prompt string.
    """
    # Build tracking context
    track_lines: List[str] = []
    for t in track_summaries:
        tid = t.get("track_id", "?")
        cls = t.get("class_name", "object")
        label = f"{cls.capitalize()} #{tid}"
        track_lines.append(f"  - {label}")

    tracking_block = ""
    if track_lines:
        tracking_block = (
            "\n\nTracked objects visible in this segment:\n"
            + "\n".join(track_lines)
        )

    prompt = (
        f"These images are {_count_word()} consecutive frames sampled from a "
        f"video segment spanning {_fmt_time(start_time)} to {_fmt_time(end_time)}.\n"
        "\n"
        "Analyze them as a temporal sequence rather than as independent images.\n"
        "\n"
        "Identify the important action or event occurring across the sequence.\n"
        f"{tracking_block}\n"
        "\n"
        "Describe:\n"
        "1. Who is involved? (use the Track IDs like 'Person #3' if available)\n"
        "2. What objects are involved?\n"
        "3. What action or event happened?\n"
        "4. What changed from the beginning of the segment to the end?\n"
        "\n"
        "Do NOT invent information that is not visible in the frames.\n"
        "Return a concise 1–3 sentence description of the meaningful event.\n"
        "If nothing notable happens, say 'No significant activity observed.'"
    )
    return prompt


SYSTEM_PROMPT = (
    "You are a precise video analysis assistant. "
    "You describe what you see in sequences of video frames. "
    "You are concise, factual, and never hallucinate. "
    "When tracking IDs are provided, use them consistently (e.g. 'Person #3')."
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS.s"""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _count_word() -> str:
    """Just returns a placeholder; actual count is set per-call."""
    return "several"
