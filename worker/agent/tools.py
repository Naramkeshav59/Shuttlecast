"""The 5 LangChain tools the ReAct agent uses per rally.

get_match_context() and generate_analysis() need running match state (score,
analysis history) that isn't part of any single tool call's arguments.
build_tools(state) binds all 5 tools to one MatchState via closure so
react_loop.py can thread state across rallies without the LLM ever seeing it
as an argument it has to (mis)manage.

Deliberate deviations from the original tool design:
  - analyze_stroke_sequence and analyze_frame_group take NO strokes/frames
    argument. Having the LLM pass that data in would mean asking the
    model to reproduce dozens of raw stroke fields (or exact file paths)
    verbatim as a tool call argument -- fragile and
    wasteful for data it never needs to see raw, only summarized. Instead
    react_loop.py calls state.set_current_rally(...) before invoking the
    agent for each rally, and every tool reads rally data from that
    closure. The same applies to assess_rally_significance and
    generate_analysis: having the model echo the observation dict back
    as JSON failed on 4 of 20 rallies. Only the agent's own decisions
    (depth, suggestion_type, the vision question) are arguments.
  - generate_analysis takes an explicit suggestion_type argument. Decision
    point 3 (suggestion type) has the *agent* choose positioning/shot_selection/
    pattern_exploitation before calling generate_analysis -- for the LLM to
    express that choice through a tool call, it has to be an argument.
"""
import os
import re
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types as genai_types
from groq import Groq
from langchain_core.tools import tool

from shared.models import AnalysisResult, Rally
from worker.agent.prompts import SUGGESTION_PROMPTS, VISION_SYSTEM_PROMPT
from worker.observability.logger import timed
from worker.pipeline.ingest import summarize_strokes

# Groq's catalog has moved on from Llama 3.3/Mixtral since this stack was
# planned; gpt-oss-120b is the current largest general-purpose model there.
# Overridable: each Groq model has its own free-tier daily quota, so a
# deployment can switch models when one is exhausted.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = "gemini-2.0-flash"


class MatchState:
    """Running state for one match's worth of rally-by-rally analysis."""

    def __init__(self, match_id: str):
        self.match_id = match_id
        self.score_a = 0
        self.score_b = 0
        self.current_rally: Rally | None = None
        self.current_frame_paths: dict[int, Path] = {}
        # keyed by (set_num, rally_id) so re-analyzing a rally (a UI
        # re-click, a re-run) replaces/no-ops instead of double-counting.
        self._scored_rallies: set[tuple[int, int]] = set()
        self._results_by_rally: dict[tuple[int, int], AnalysisResult] = {}

    @property
    def analysis_history(self) -> list[AnalysisResult]:
        return list(self._results_by_rally.values())

    def set_current_rally(self, rally: Rally, frame_paths: dict[int, Path]) -> None:
        self.current_rally = rally
        self.current_frame_paths = frame_paths

    def record_rally_outcome(self, set_num: int, rally_id: int, rally_winner: str | None) -> None:
        key = (set_num, rally_id)
        if key in self._scored_rallies:
            return
        self._scored_rallies.add(key)
        if rally_winner == "A":
            self.score_a += 1
        elif rally_winner == "B":
            self.score_b += 1

    def record_analysis(self, result: AnalysisResult) -> None:
        self._results_by_rally[(result.set_num, result.rally_id)] = result


def _parse_pattern_suggestion(text: str) -> tuple[str, str]:
    pattern_match = re.search(r"PATTERN:\s*(.+?)(?:\nSUGGESTION:|$)", text, re.DOTALL)
    suggestion_match = re.search(r"SUGGESTION:\s*(.+)", text, re.DOTALL)
    tactical_pattern = pattern_match.group(1).strip() if pattern_match else text.strip()
    suggestion_text = suggestion_match.group(1).strip() if suggestion_match else ""
    return tactical_pattern, suggestion_text


def build_generation_prompt(observation: dict, context: dict, depth: str, suggestion_type: str) -> str:
    """Without the explicit winner/loser line, the model critiqued player B
    even in rallies B won (tactical accuracy 62.5% on 20 rallies)."""
    winner = observation.get("rally_winner")
    if winner in ("A", "B"):
        loser = "B" if winner == "A" else "A"
        outcome = (f"Outcome: player {winner} won this rally, so the decisive error belongs to "
                   f"player {loser}. Analyze player {loser}'s play, and name player {loser} in the PATTERN.")
    else:
        # uploaded clips: strokes come from the vision models and the winner isn't known
        outcome = ("Outcome: the winner of this rally isn't known. Identify the decisive stroke from "
                   "the sequence, and name the player who played it in the PATTERN.")
    return SUGGESTION_PROMPTS[suggestion_type].format(
        observation=observation, context=context, depth=depth, outcome=outcome,
    )


def build_tools(state: MatchState) -> list:
    """Return the 5 tools bound to `state`, ready to hand to a LangChain agent."""

    @tool
    def analyze_frame_group(question: str) -> str:
        """Ask Gemini Flash a specific analytical question about the current
        rally's stroke-timestamp frames."""
        if not os.environ.get("GEMINI_API_KEY"):
            return "Vision analysis unavailable: GEMINI_API_KEY is not configured."
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        parts = [genai_types.Part.from_text(text=f"{VISION_SYSTEM_PROMPT}\n\nQuestion: {question}")]
        for path in state.current_frame_paths.values():
            parts.append(genai_types.Part.from_bytes(
                data=Path(path).read_bytes(), mime_type="image/jpeg",
            ))
        with timed("vision_gemini", n_frames=len(parts) - 1):
            response = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
        return response.text

    # Rally data is read from `state`, never passed through tool arguments:
    # making the model re-emit the stroke dict as JSON failed on 4 of 20
    # rallies (Groq 400 "failed to parse tool call arguments"). Tool
    # arguments carry only what the agent decides.
    def _observation() -> dict:
        return summarize_strokes(state.current_rally.strokes if state.current_rally else [])

    def _match_context() -> dict:
        return {
            "match_id": state.match_id,
            "score_a": state.score_a,
            "score_b": state.score_b,
            "rallies_analyzed": len(state.analysis_history),
            "recent_suggestions": [
                r.suggestion_text for r in state.analysis_history[-3:] if r.suggestion_text
            ],
        }

    @tool
    def analyze_stroke_sequence() -> dict:
        """Summarize ShuttleSet stroke metadata for the current rally: shot
        sequence, positions, landing zones, and rally outcome."""
        return _observation()

    @tool
    def assess_rally_significance() -> str:
        """Decide if the current rally warrants deep analysis or a brief note.
        Returns 'deep' | 'surface' | 'skip'. A first-pass heuristic the
        agent's own reasoning can accept or override in later Thought steps."""
        obs = _observation()
        rally_length = obs["rally_length"]
        shot_sequence = obs["shot_sequence"]
        if rally_length <= 3:
            return "skip"
        if len(shot_sequence) >= 2 and shot_sequence[-1] == shot_sequence[-2]:
            return "deep"
        if rally_length >= 8:
            return "deep"
        return "surface"

    @tool
    def get_match_context() -> dict:
        """Return current score and recent analysis history for the match
        being processed."""
        return _match_context()

    @tool
    def generate_analysis(
        depth: Literal["deep", "surface", "skip"],
        suggestion_type: Literal["positioning", "shot_selection", "pattern_exploitation"],
    ) -> str:
        """Call Groq to generate tactical analysis + improvement suggestion
        for the current rally at the given depth and suggestion_type."""
        prompt = build_generation_prompt(_observation(), _match_context(), depth, suggestion_type)
        # this call shares the same account-wide 8000 TPM cap (on Groq's free
        # on_demand tier) as the ReAct loop's own LLM calls, so it hits 429s
        # too; the SDK's own max_retries (default 2) backs off automatically.
        client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=6)
        with timed("generate_groq", suggestion_type=suggestion_type, depth=depth):
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        return response.choices[0].message.content

    return [
        analyze_frame_group,
        analyze_stroke_sequence,
        assess_rally_significance,
        get_match_context,
        generate_analysis,
    ]
