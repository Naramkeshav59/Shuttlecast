"""Chat Q&A over a finished match analysis.

Answers are grounded only in the per-rally analyses already produced (the
JobResult), with rally citations -- the model never sees raw video or
invents rallies. Lives in ui/ so the AWS UI container needs no worker code.
"""
import os

from groq import Groq

from shared.models import JobResult

QA_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are ShuttleCast, a badminton tactics assistant. Answer \
the user's question using ONLY the per-rally analyses of this match given \
below. Cite the rallies you rely on like (Rally 7). If the analyses don't \
cover the question, say so plainly instead of guessing. Player A and player B \
are the two players; keep those labels. Answer in under 120 words."""


def _rally_lines(result: JobResult) -> str:
    lines = []
    for r in result.rallies:
        a = r.analysis
        head = (f"Rally {a.rally_id} (set {a.set_num}, won by {r.rally_winner}, "
                f"score after {r.score_a}-{r.score_b}, {r.n_strokes} strokes, depth {a.depth})")
        if a.depth == "skip":
            lines.append(f"{head}: no coaching note.")
        else:
            lines.append(f"{head} [{a.suggestion_type}]: {a.tactical_pattern} Suggestion: {a.suggestion_text}")
    return "\n".join(lines)


def answer(question: str, result: JobResult, history: list[dict]) -> str:
    """history: prior chat turns as {"role": "user"|"assistant", "content": str}."""
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nMatch: {result.match_id}\n\n{_rally_lines(result)}"},
        *history[-6:],  # enough for follow-ups without resending a long chat
        {"role": "user", "content": question},
    ]
    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=3)
    return client.chat.completions.create(model=QA_MODEL, messages=messages).choices[0].message.content
