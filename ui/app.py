"""Streamlit UI for ShuttleCast.

Two modes:
  * AWS mode (SHUTTLECAST_API_URL set): submit a job to API Gateway, poll
    the status Lambda, then render the JobResult from its presigned S3 URL
    synced to the YouTube video. Needs only shared/ + data/*.json/csv --
    never imports worker code, so the UI container stays small.
  * Local mode (no API URL): run the agent rally-by-rally in-process, or
    load a full-match result written by scripts/local_run.py.
"""
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from shared.models import JobResult  # noqa: E402

API_URL = os.environ.get("SHUTTLECAST_API_URL", "").rstrip("/")
REGISTRY = REPO_ROOT / "data" / "verified_videos.json"
MATCH_CSV = REPO_ROOT / "data" / "shuttleset" / "set" / "match.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results"

st.set_page_config(page_title="ShuttleCast", layout="wide")
st.title("ShuttleCast — Per-Rally Tactical Analysis")


@st.cache_data
def verified_match_urls() -> dict[str, str]:
    """match_id -> YouTube URL, for matches whose clip is verified aligned."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    with MATCH_CSV.open(encoding="utf-8") as f:
        urls = {row["video"]: row["url"] for row in csv.DictReader(f) if row.get("url")}
    return {
        m: urls[m] for m, entry in registry.items()
        if not m.startswith("_") and entry.get("aligned") and m in urls
    }


def render_trace(messages: list) -> None:
    with st.expander("Agent trace (Thought / Action / Observation)"):
        for msg in messages:
            role = msg.__class__.__name__
            if role == "HumanMessage":
                st.text(f"[Question] {msg.content}")
            elif role == "AIMessage":
                if msg.content:
                    st.text(f"[Thought] {msg.content}")
                for call in getattr(msg, "tool_calls", None) or []:
                    st.text(f"[Action] {call['name']}({call['args']})")
            elif role == "ToolMessage":
                st.text(f"[Observation] {msg.content}")


def render_analysis(analysis) -> None:
    st.markdown(f"**Depth:** `{analysis.depth}`")
    if analysis.depth != "skip":
        st.markdown(f"**Suggestion type:** `{analysis.suggestion_type}`")
        st.markdown(f"**Tactical pattern:** {analysis.tactical_pattern}")
        st.markdown(f"**Suggestion:** {analysis.suggestion_text}")


def render_job_result(result: JobResult, video_source: str) -> None:
    """Full-match viewer, shared by AWS mode and saved local results."""
    if not result.rallies:
        st.warning("No rallies were analyzed.")
        return
    depths = [r.analysis.depth for r in result.rallies]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rallies analyzed", len(result.rallies))
    c2.metric("Deep", depths.count("deep"))
    c3.metric("Surface", depths.count("surface"))
    c4.metric("Skipped", depths.count("skip"))
    if result.errors:
        st.warning(f"{len(result.errors)} rallies failed: "
                   + ", ".join(f"set {e.set_num} rally {e.rally_id}" for e in result.errors))

    col_list, col_main = st.columns([1, 2])
    with col_list:
        idx = st.radio(
            "Rally", range(len(result.rallies)), key="result_rally",
            format_func=lambda i: (
                f"Set {result.rallies[i].analysis.set_num} / Rally {result.rallies[i].analysis.rally_id}"
                f" — {result.rallies[i].analysis.depth}"
            ),
        )
    report = result.rallies[idx]
    with col_main:
        st.subheader(f"Set {report.analysis.set_num} / Rally {report.analysis.rally_id}")
        st.video(video_source, start_time=int(report.start_time_sec))
        st.metric("Score after this rally", f"{report.score_a} - {report.score_b}")
        st.caption(f"{report.n_strokes} strokes, won by {report.rally_winner}")
        render_analysis(report.analysis)


# ---- AWS mode ----------------------------------------------------------

def aws_mode() -> None:
    st.caption(f"Connected to {API_URL}")
    matches = verified_match_urls()
    with st.form("submit"):
        match_id = st.selectbox("Match", list(matches))
        submitted = st.form_submit_button("Analyze match")
    if submitted:
        resp = requests.post(
            f"{API_URL}/jobs", json={"match_id": match_id, "youtube_url": matches[match_id]}, timeout=15,
        )
        if resp.status_code != 202:
            st.error(f"Submit failed ({resp.status_code}): {resp.text}")
            return
        st.session_state.job_id = resp.json()["job_id"]
        st.session_state.pop("job_result", None)
        st.query_params["job"] = st.session_state.job_id

    # a job_id in the URL survives a page refresh
    job_id = st.session_state.get("job_id") or st.query_params.get("job")
    if not job_id:
        return
    st.session_state.job_id = job_id

    if "job_result" in st.session_state:
        result, youtube_url = st.session_state.job_result
        render_job_result(result, youtube_url)
        return
    poll_job(job_id)


@st.fragment(run_every=5)
def poll_job(job_id: str) -> None:
    resp = requests.get(f"{API_URL}/jobs/{job_id}", timeout=15)
    if resp.status_code != 200:
        st.error(f"Status check failed ({resp.status_code}): {resp.text}")
        return
    job = resp.json()
    status = job["status"]
    if status == "complete":
        result = JobResult(**requests.get(job["result_url"], timeout=30).json())
        st.session_state.job_result = (result, job["youtube_url"])
        st.rerun(scope="app")
    elif status == "failed":
        st.error(f"Job failed: {job.get('error')}")
    else:
        note = f" — {job['error']}" if job.get("error") else ""
        st.info(f"Job `{job_id}` is **{status}**{note}. Checking every 5s…")


# ---- local mode ----------------------------------------------------------

def local_mode() -> None:
    from worker.agent.react_loop import analyze_rally, build_agent
    from worker.agent.tools import MatchState
    from worker.pipeline.frame_extractor import get_video_fps
    from worker.pipeline.ingest import load_match, verified_videos

    st.caption("Local mode — set SHUTTLECAST_API_URL to use the AWS backend.")
    videos = verified_videos()
    if not videos:
        st.error("No verified clips in data/videos/ — see data/verified_videos.json.")
        return
    match_id = st.selectbox("Match", list(videos))
    video_path = videos[match_id]

    saved = RESULTS_DIR / f"{match_id}.json"
    view = "Interactive (per rally)"
    if saved.exists():
        view = st.radio("View", ["Saved full-match result", "Interactive (per rally)"], horizontal=True)
    if view == "Saved full-match result":
        render_job_result(JobResult(**json.loads(saved.read_text(encoding="utf-8"))), str(video_path))
        return

    # Match state lives in st.session_state, NOT @st.cache_resource: that
    # cache is process-global, so it would share one score across every
    # browser session and double-count on reload.
    if st.session_state.get("match_id") != match_id:
        st.session_state.match_id = match_id
        st.session_state.match_state = MatchState(match_id=match_id)
        st.session_state.agent = build_agent(st.session_state.match_state)
        st.session_state.results = {}
        st.session_state.traces = {}
    state = st.session_state.match_state
    agent = st.session_state.agent
    rallies = load_match(match_id)
    fps = get_video_fps(video_path)

    col_list, col_main = st.columns([1, 2])
    with col_list:
        st.subheader(f"Rallies ({len(rallies)})")

        def _label(i: int) -> str:
            r = rallies[i]
            done = "done" if (r.set_num, r.rally_id) in st.session_state.results else "pending"
            return f"Set {r.set_num} / Rally {r.rally_id} — {len(r.strokes)} strokes, won by {r.rally_winner} [{done}]"

        selected_idx = st.radio(
            "Select a rally", range(len(rallies)), format_func=_label,
            label_visibility="collapsed", key="rally_selector",
        )

    rally = rallies[selected_idx]
    key = (rally.set_num, rally.rally_id)
    with col_main:
        st.subheader(f"Set {rally.set_num} / Rally {rally.rally_id}")
        start_time = int(rally.strokes[0].frame_num / fps) if rally.strokes else 0
        st.video(str(video_path), start_time=start_time)

        if st.button("Analyze this rally"):
            from groq import RateLimitError

            try:
                with st.spinner("Running ReAct agent (analyze_stroke_sequence -> ... -> generate_analysis)..."):
                    result, messages = analyze_rally(agent, state, rally, video_path)
            except RateLimitError:
                # The 8000 tokens/min cap is per Groq account, not per process --
                # a concurrent run (e.g. training/prepare_dataset.py) can starve
                # this one past all its retries.
                st.error(
                    "Groq rate limit reached (8,000 tokens/min for the whole account). "
                    "If another job is using the same key — e.g. training/prepare_dataset.py — "
                    "wait for it to finish, or try again in a minute."
                )
            else:
                st.session_state.results[key] = result
                st.session_state.traces[key] = messages
                # the sidebar label was drawn earlier in this same script pass,
                # so it's stale until the next rerun -- force one now.
                st.rerun()

        if key in st.session_state.results:
            # only counts rallies analyzed so far, in whatever order they were clicked
            st.metric("Score (analyzed rallies)", f"{state.score_a} - {state.score_b}")
            render_analysis(st.session_state.results[key])
            render_trace(st.session_state.traces[key])
        else:
            st.info("Click \"Analyze this rally\" to run the agent on it.")


if API_URL:
    aws_mode()
else:
    local_mode()
