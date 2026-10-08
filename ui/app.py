"""ShuttleCast UI: a chat-style match analyst.

Each analyzed rally is a message from the agent; the video panel stays
pinned and jumps to whichever rally you pick; the chat box answers
questions grounded in the analyses, or ("analyze rally N") runs the agent
live in local mode.

Modes:
  * AWS mode (SHUTTLECAST_API_URL set): submit a job to API Gateway, poll
    the status Lambda, load the JobResult from its presigned S3 URL. Never
    imports worker code, so the UI container stays small.
  * Local mode: load results written by scripts/local_run.py and run the
    agent in-process for individual rallies.
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "ui"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from match_qa import answer  # noqa: E402
from shared.models import JobResult, RallyReport  # noqa: E402

API_URL = os.environ.get("SHUTTLECAST_API_URL", "").rstrip("/")
REGISTRY = REPO_ROOT / "data" / "verified_videos.json"
MATCH_CSV = REPO_ROOT / "data" / "shuttleset" / "set" / "match.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results"
ANALYZE_CMD = re.compile(r"^\s*analy[sz]e\s+rally\s+(\d+)(?:\s+(?:of\s+)?set\s+(\d+))?\s*$", re.I)

st.set_page_config(page_title="ShuttleCast", page_icon="🏸", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
[data-testid="stChatMessage"] {border: 1px solid rgba(128,128,128,.18); border-radius: 14px; padding: .8rem 1rem;}
.sc-pill {display:inline-block; padding:1px 10px; border-radius:999px; font-size:.72rem; font-weight:600;
          letter-spacing:.02em; margin-right:6px; text-transform:uppercase; vertical-align:middle;}
.sc-deep {background:#fde2e1; color:#b42318;}
.sc-surface {background:#fef0c7; color:#b54708;}
.sc-skip {background:#eaecf0; color:#475467;}
.sc-type {background:#e0eaff; color:#3538cd;}
.sc-meta {color:#667085; font-size:.84rem;}
.sc-label {font-weight:600; color:#203a43;}
/* keep the video panel in view while the conversation scrolls */
div[data-testid="stColumn"]:nth-of-type(2) {position: sticky; top: 3.5rem; align-self: flex-start;}
</style>
""", unsafe_allow_html=True)


# ---- data -----------------------------------------------------------------

@st.cache_data
def match_index() -> dict[str, dict]:
    with MATCH_CSV.open(encoding="utf-8") as f:
        return {row["video"]: row for row in csv.DictReader(f)}


@st.cache_data
def verified_matches() -> dict[str, dict]:
    """match_id -> {"url": YouTube URL, "file": local clip name}."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    index = match_index()
    return {
        m: {"url": index.get(m, {}).get("url") or None, "file": entry["file"]}
        for m, entry in registry.items()
        if not m.startswith("_") and entry.get("aligned")
    }


def short_name(match_id: str) -> str:
    row = match_index().get(match_id)
    if not row:
        return match_id.replace("_", " ")
    return f"{row['winner']} vs {row['loser']} · {row['tournament']} {row['round']}"


def load_saved(match_id: str) -> JobResult | None:
    path = RESULTS_DIR / f"{match_id}.json"
    return JobResult(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None


# ---- rendering --------------------------------------------------------------

def pills(analysis) -> str:
    html = f'<span class="sc-pill sc-{analysis.depth}">{analysis.depth}</span>'
    if analysis.suggestion_type:
        html += f'<span class="sc-pill sc-type">{analysis.suggestion_type.replace("_", " ")}</span>'
    return html


def render_trace(messages: list) -> None:
    with st.expander("How the agent decided"):
        for msg in messages:
            role = msg.__class__.__name__
            if role == "AIMessage":
                if msg.content:
                    st.markdown(f"**Thought:** {msg.content}")
                for call in getattr(msg, "tool_calls", None) or []:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    st.markdown(f"**Action:** `{call['name']}({args})`")
            elif role == "ToolMessage":
                st.code(str(msg.content)[:600], language=None)


def render_rally(report: RallyReport, trace: list | None = None) -> None:
    a = report.analysis
    key = (a.set_num, a.rally_id)
    with st.chat_message("assistant", avatar="🏸"):
        head, btn = st.columns([9, 1])
        head.markdown(
            f"**Set {a.set_num} · Rally {a.rally_id}** &nbsp;{pills(a)}<br>"
            f"<span class='sc-meta'>Won by player {report.rally_winner} · {report.n_strokes} strokes · "
            f"score {report.score_a}–{report.score_b}</span>",
            unsafe_allow_html=True,
        )
        if btn.button("▶", key=f"watch-{key}", help="Watch this rally", use_container_width=True):
            st.session_state.now_playing = key
        if a.depth == "skip":
            st.markdown("<span class='sc-meta'>Clean rally — nothing worth coaching.</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='sc-label'>What happened</span> — {a.tactical_pattern}", unsafe_allow_html=True)
            st.markdown(f"<span class='sc-label'>Try instead</span> — {a.suggestion_text}", unsafe_allow_html=True)
        if trace:
            render_trace(trace)


def render_sidebar_stats(result: JobResult) -> list[str]:
    depths = [r.analysis.depth for r in result.rallies]
    last = result.rallies[-1]
    st.sidebar.markdown("#### Match so far")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Score (A–B)", f"{last.score_a}–{last.score_b}")
    c2.metric("Rallies", len(result.rallies))
    c1, c2, c3 = st.sidebar.columns(3)
    c1.metric("Deep", depths.count("deep"))
    c2.metric("Surface", depths.count("surface"))
    c3.metric("Skip", depths.count("skip"))
    if result.errors:
        st.sidebar.warning(f"{len(result.errors)} rallies failed to analyze")
    return st.sidebar.multiselect(
        "Show", ["deep", "surface", "skip"], default=["deep", "surface", "skip"],
    )


def render_video_panel(result: JobResult, source: str) -> None:
    by_key = {(r.analysis.set_num, r.analysis.rally_id): r for r in result.rallies}
    key = st.session_state.get("now_playing")
    report = by_key.get(key) or result.rallies[0]
    a = report.analysis
    st.markdown(f"**Now playing** · Set {a.set_num} · Rally {a.rally_id} &nbsp;{pills(a)}", unsafe_allow_html=True)
    st.video(source, start_time=int(report.start_time_sec))
    st.caption(f"Jumps to the rally's first stroke ({int(report.start_time_sec // 60)}:{int(report.start_time_sec % 60):02d}). "
               "Press ▶ on any rally to switch.")


# ---- chat ------------------------------------------------------------------

def run_live_analysis(match_id: str, set_num: int, rally_id: int, video: Path, result: JobResult) -> JobResult:
    """Local mode only: run the agent on one rally and merge it into the result."""
    from worker.agent.react_loop import analyze_match

    traces: dict = {}
    fresh = analyze_match(match_id, video, only={(set_num, rally_id)}, traces=traces)
    if fresh.errors:
        raise RuntimeError(fresh.errors[0].error)
    st.session_state.setdefault("traces", {}).update(traces)
    keep = [r for r in result.rallies if (r.analysis.set_num, r.analysis.rally_id) != (set_num, rally_id)]
    merged = sorted(keep + fresh.rallies, key=lambda r: (r.analysis.set_num, r.analysis.rally_id))
    return result.model_copy(update={"rallies": merged})


def handle_prompt(prompt: str, result: JobResult, match_id: str, video: Path | None) -> JobResult:
    from groq import RateLimitError

    chat = st.session_state.chat
    chat.append({"role": "user", "content": prompt})
    cmd = ANALYZE_CMD.match(prompt)
    try:
        if cmd:
            if video is None:
                chat.append({"role": "assistant", "content": "Live analysis runs in local mode only; "
                             "in AWS mode the worker analyzes the whole match."})
                return result
            rally_id, set_num = int(cmd.group(1)), int(cmd.group(2) or 1)
            with st.spinner(f"Agent analyzing set {set_num}, rally {rally_id}…"):
                result = run_live_analysis(match_id, set_num, rally_id, video, result)
            st.session_state.now_playing = (set_num, rally_id)
            chat.append({"role": "assistant", "content": f"Done — set {set_num}, rally {rally_id} is in the "
                         "feed above with the agent's reasoning trace, and the video jumped to it."})
        else:
            with st.spinner("Thinking…"):
                reply = answer(prompt, result, chat[:-1])
            chat.append({"role": "assistant", "content": reply})
    except RateLimitError:
        chat.append({"role": "assistant", "content": "I've hit Groq's rate limit (shared with any other job "
                     "using the same key). Give it a minute and ask again."})
    except Exception as exc:  # surface failures in the chat instead of a stack trace
        chat.append({"role": "assistant", "content": f"That didn't work: {exc}"})
    return result


def conversation(result: JobResult, match_id: str, video: Path | None, source: str) -> None:
    st.session_state.setdefault("chat", [])
    shown = render_sidebar_stats(result)

    feed, side = st.columns([3, 2], gap="large")
    with feed:
        st.markdown(f"### {short_name(match_id)}")
        st.caption("Each message is the agent's take on one rally. Ask a question below, "
                   "or type **analyze rally 21** to watch the agent work live.")
        traces = st.session_state.get("traces", {})
        for report in result.rallies:
            if report.analysis.depth in shown:
                render_rally(report, traces.get((report.analysis.set_num, report.analysis.rally_id)))
        for turn in st.session_state.chat:
            with st.chat_message(turn["role"], avatar="🏸" if turn["role"] == "assistant" else "🙂"):
                st.markdown(turn["content"])
        if not st.session_state.chat:
            st.markdown("<span class='sc-meta'>Try asking:</span>", unsafe_allow_html=True)
            ideas = ["Where did B lose points?", "Summarize the match", "What should A change?"]
            for col, idea in zip(st.columns(len(ideas)), ideas):
                if col.button(idea, use_container_width=True):
                    st.session_state.pending_prompt = idea
    with side:
        render_video_panel(result, source)

    prompt = st.chat_input("Ask about this match, or type 'analyze rally 21'")
    prompt = prompt or st.session_state.pop("pending_prompt", None)
    if prompt:
        st.session_state.result = handle_prompt(prompt, result, match_id, video)
        st.rerun()


# ---- modes -----------------------------------------------------------------

def sidebar_header() -> None:
    st.sidebar.markdown("## 🏸 ShuttleCast")
    st.sidebar.caption("An AI agent that coaches every rally of a pro badminton match.")


def pick_match(options: list[str]) -> str:
    match_id = st.sidebar.selectbox("Match", options, format_func=short_name)
    if st.session_state.get("match_id") != match_id:
        for k in ("result", "chat", "traces", "now_playing", "job_id"):
            st.session_state.pop(k, None)
        st.session_state.match_id = match_id
    return match_id


def local_mode() -> None:
    sidebar_header()
    matches = verified_matches()
    match_id = pick_match(list(matches))
    video = REPO_ROOT / "data" / "videos" / matches[match_id]["file"]
    if "result" not in st.session_state:
        st.session_state.result = load_saved(match_id)
    result = st.session_state.result
    st.sidebar.caption("Local mode · agent runs on this machine")
    if result is None or not result.rallies:
        st.info("No saved analysis for this match yet. Run "
                f"`python3 scripts/local_run.py --match-id {match_id} --max-rallies 10`, then reload.")
        return
    conversation(result, match_id, video, str(video))


def aws_mode() -> None:
    sidebar_header()
    matches = {m: v for m, v in verified_matches().items() if v["url"]}
    match_id = pick_match(list(matches))
    st.sidebar.caption("Connected to the AWS backend")
    if st.sidebar.button("Analyze this match", type="primary", use_container_width=True):
        resp = requests.post(f"{API_URL}/jobs", json={"match_id": match_id, "youtube_url": matches[match_id]["url"]},
                             timeout=15)
        if resp.status_code != 202:
            st.sidebar.error(f"Submit failed ({resp.status_code}): {resp.text}")
        else:
            st.session_state.job_id = resp.json()["job_id"]
            st.query_params["job"] = st.session_state.job_id

    job_id = st.session_state.get("job_id") or st.query_params.get("job")
    if "result" in st.session_state:
        conversation(st.session_state.result, match_id, None, matches[match_id]["url"])
    elif job_id:
        st.session_state.job_id = job_id
        poll_job(job_id)
    else:
        st.info("Pick a match and press **Analyze this match**. The worker analyzes every rally "
                "asynchronously; results appear here as a conversation.")


@st.fragment(run_every=5)
def poll_job(job_id: str) -> None:
    resp = requests.get(f"{API_URL}/jobs/{job_id}", timeout=15)
    if resp.status_code != 200:
        st.error(f"Status check failed ({resp.status_code}): {resp.text}")
        return
    job = resp.json()
    if job["status"] == "complete":
        st.session_state.result = JobResult(**requests.get(job["result_url"], timeout=30).json())
        st.rerun(scope="app")
    elif job["status"] == "failed":
        st.error(f"Job failed: {job.get('error')}")
    else:
        note = f" — {job['error']}" if job.get("error") else ""
        with st.chat_message("assistant", avatar="🏸"):
            st.markdown(f"Working on it — job is **{job['status']}**{note}. This page updates by itself.")


if API_URL:
    aws_mode()
else:
    local_mode()
