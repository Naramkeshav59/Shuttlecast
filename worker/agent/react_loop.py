"""Wire the 5 tools into a LangChain ReAct agent and run it per rally.

Uses langgraph.prebuilt.create_react_agent rather than the older
langchain.agents.create_react_agent + AgentExecutor combo. The classic
AgentExecutor ReAct pattern has the model emit a single free-text "Action
Input" string per tool call, which can't represent our tools' real argument
shapes (dicts, literals). langgraph's create_react_agent drives the same
think -> act -> observe loop but through the model's native tool-calling
(the model emits structured JSON per argument), which is what
assess_rally_significance/generate_analysis's dict/Literal arguments need,
and is also the actively-maintained implementation as of late 2025.
"""
import os
from collections.abc import Callable
from pathlib import Path

from langchain_groq import ChatGroq

from shared.models import AnalysisResult, JobResult, Rally, RallyError, RallyReport
from worker.agent.prompts import REACT_SYSTEM_PROMPT
from worker.agent.tools import GROQ_MODEL, MatchState, _parse_pattern_suggestion, build_tools
from worker.observability.logger import log_event, timed
from worker.pipeline.frame_extractor import check_fps_alignment, extract_rally_frames
from worker.pipeline.ingest import expected_fps, load_match


def build_agent(state: MatchState):
    from langgraph.prebuilt import create_react_agent as _create_react_agent

    # Groq's free on_demand tier caps gpt-oss-120b at 8000 tokens/minute; a
    # multi-step ReAct loop (one LLM call per Thought, plus tool schemas and
    # the system prompt resent every call) burns through that within 1-2
    # rallies. max_retries lets LangChain's built-in backoff ride out 429s
    # instead of the whole run dying.
    llm = ChatGroq(model=GROQ_MODEL, api_key=os.environ["GROQ_API_KEY"], temperature=0, max_retries=6)
    tools = build_tools(state)
    return _create_react_agent(llm, tools, prompt=REACT_SYSTEM_PROMPT)


def _generate_analysis_call(messages: list) -> dict | None:
    """Find the args of the (last) generate_analysis tool call, if the agent
    made one -- that's where depth/suggestion_type actually live, since the
    final message is just its return value, not its own arguments."""
    for msg in reversed(messages):
        for call in getattr(msg, "tool_calls", None) or []:
            if call["name"] == "generate_analysis":
                return call["args"]
    return None


def analyze_rally(agent, state: MatchState, rally: Rally, video_path: Path) -> tuple[AnalysisResult, list]:
    """Run the agent on one rally. Returns the resulting AnalysisResult
    (recorded into `state`) plus the full message trace for inspection."""
    with timed("frame_extraction", set_num=rally.set_num, rally_id=rally.rally_id):
        frame_paths = extract_rally_frames(video_path, rally)
    state.set_current_rally(rally, frame_paths)

    task = (
        f"Analyze rally {rally.rally_id} of set {rally.set_num} "
        f"(match {rally.match_id}). {len(rally.strokes)} strokes were played; "
        f"{len(frame_paths)} broadcast frames were extracted at each stroke's "
        f"timestamp for visual analysis. "
        + (f"The rally was won by player {rally.rally_winner}." if rally.rally_winner
           else "The rally's winner isn't known (strokes were detected from video).")
    )
    output = agent.invoke({"messages": [{"role": "user", "content": task}]})
    messages = output["messages"]
    state.record_rally_outcome(rally.set_num, rally.rally_id, rally.rally_winner)

    call_args = _generate_analysis_call(messages)
    if call_args is None:
        # assess_rally_significance returned 'skip' (or the agent never got
        # that far) -- no generate_analysis call means no suggestion to record.
        result = AnalysisResult(
            match_id=rally.match_id, set_num=rally.set_num, rally_id=rally.rally_id,
            depth="skip",
        )
    else:
        tactical_pattern, suggestion_text = _parse_pattern_suggestion(messages[-1].content)
        result = AnalysisResult(
            match_id=rally.match_id, set_num=rally.set_num, rally_id=rally.rally_id,
            depth=call_args.get("depth", "surface"),
            tactical_pattern=tactical_pattern,
            suggestion_type=call_args.get("suggestion_type"),
            suggestion_text=suggestion_text,
        )
    state.record_analysis(result)
    return result, messages


def analyze_match(
    match_id: str,
    video_path: Path,
    *,
    max_rallies: int | None = None,
    job_id: str | None = None,
    youtube_url: str | None = None,
    on_rally_done: Callable[[Rally], None] | None = None,
    on_progress: Callable[[JobResult, int], None] | None = None,
    only: set[tuple[int, int]] | None = None,
    traces: dict | None = None,
) -> JobResult:
    """Run the agent over every rally of a match. Shared by the AWS worker
    (worker/main.py) and scripts/local_run.py so there's one code path.

    `only` limits agent runs to those (set_num, rally_id) pairs; every other
    rally's outcome is still replayed so running scores stay correct.
    Pass a dict as `traces` to collect each analyzed rally's message trace."""
    rallies = load_match(match_id)
    if max_rallies:
        rallies = rallies[:max_rallies]
    fps = check_fps_alignment(video_path, expected_fps(match_id))
    return analyze_rallies(
        match_id, rallies, video_path, fps, job_id=job_id, youtube_url=youtube_url,
        on_rally_done=on_rally_done, on_progress=on_progress, only=only, traces=traces,
    )


def analyze_rallies(
    match_id: str,
    rallies: list[Rally],
    video_path: Path,
    fps: float,
    *,
    job_id: str | None = None,
    youtube_url: str | None = None,
    on_rally_done: Callable[[Rally], None] | None = None,
    on_progress: Callable[[JobResult, int], None] | None = None,
    only: set[tuple[int, int]] | None = None,
    traces: dict | None = None,
) -> JobResult:
    """The per-rally agent loop, for rallies from any source: ShuttleSet
    annotations (analyze_match) or strokes detected in an uploaded video
    (vision/infer.py)."""
    state = MatchState(match_id=match_id)
    agent = build_agent(state)
    reports: list[RallyReport] = []
    errors: list[RallyError] = []
    todo = sum(1 for r in rallies if only is None or (r.set_num, r.rally_id) in only)

    def snapshot() -> JobResult:
        return JobResult(job_id=job_id, match_id=match_id, youtube_url=youtube_url, fps=fps,
                         rallies=list(reports), errors=list(errors))

    for rally in rallies:
        if only is not None and (rally.set_num, rally.rally_id) not in only:
            state.record_rally_outcome(rally.set_num, rally.rally_id, rally.rally_winner)
            continue
        try:
            with timed("agent_rally", set_num=rally.set_num, rally_id=rally.rally_id):
                try:
                    result, messages = analyze_rally(agent, state, rally, video_path)
                except Exception as exc:
                    # gpt-oss on Groq sometimes writes its reasoning where the
                    # tool call belongs (400 output_parse_failed). It's a sampling
                    # fluke, not a bad input, so one retry usually succeeds.
                    if "output_parse_failed" not in str(exc):
                        raise
                    log_event("rally_retry", set_num=rally.set_num, rally_id=rally.rally_id)
                    result, messages = analyze_rally(agent, state, rally, video_path)
            if traces is not None:
                traces[(rally.set_num, rally.rally_id)] = messages
            reports.append(RallyReport(
                analysis=result,
                start_time_sec=rally.strokes[0].frame_num / fps if rally.strokes else 0.0,
                rally_winner=rally.rally_winner,
                n_strokes=len(rally.strokes),
                score_a=state.score_a,
                score_b=state.score_b,
            ))
        except Exception as exc:
            # One bad rally (a malformed tool call, a 429 that outlived the
            # retries) shouldn't sink a 75-rally job. Keep the running score
            # right regardless -- record_rally_outcome is idempotent.
            state.record_rally_outcome(rally.set_num, rally.rally_id, rally.rally_winner)
            errors.append(RallyError(set_num=rally.set_num, rally_id=rally.rally_id, error=repr(exc)[:500]))
            log_event("rally_failed", set_num=rally.set_num, rally_id=rally.rally_id, error=repr(exc)[:500])
        if on_rally_done:
            on_rally_done(rally)
        if on_progress:
            on_progress(snapshot(), todo)

    return snapshot()


def print_trace(messages: list) -> None:
    """Render the message list as a readable Thought/Action/Observation trace."""
    for msg in messages:
        role = msg.__class__.__name__
        if role == "HumanMessage":
            print(f"[Question] {msg.content}")
        elif role == "AIMessage":
            if msg.content:
                print(f"[Thought] {msg.content}")
            for call in getattr(msg, "tool_calls", None) or []:
                print(f"[Action] {call['name']}({call['args']})")
        elif role == "ToolMessage":
            print(f"[Observation] {msg.content}")
        print()
