"""Record a demo of the live deployment with a scripted browser.

Drives the real AWS UI (real jobs, real agent), records the browser to
webm, and logs when each scene starts so scripts/edit_demo.py can speed up
the waiting and keep the interactions at 1x.

    python scripts/record_demo.py <ui_url> <video_to_upload> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

UI, UPLOAD, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SIZE = {"width": 1440, "height": 900}
T0 = 0.0  # set when the page (and its recording) starts
scenes: list[dict] = []

CAPTION_JS = """(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:99999;' +
      'background:rgba(17,24,39,.92);color:#fff;font:600 22px/1.35 system-ui,Segoe UI,sans-serif;' +
      'padding:14px 26px;border-radius:12px;max-width:1100px;text-align:center;box-shadow:0 6px 24px rgba(0,0,0,.25)';
    document.body.appendChild(el);
  }
  el.style.display = text ? 'block' : 'none';
  el.textContent = text;
}"""


def scene(page: Page, name: str, caption: str, speed: float = 1.0) -> None:
    """Mark a scene boundary; speed > 1 means the editor fast-forwards it."""
    scenes.append({"name": name, "t": time.monotonic() - T0, "speed": speed})
    label = f"{caption}   ⏩ {speed:g}× speed" if speed > 1 else caption
    page.evaluate(CAPTION_JS, label)


def keep_caption(page: Page, caption: str) -> None:
    # Streamlit reruns can drop body children; re-assert without a new scene
    page.evaluate(CAPTION_JS, caption)


def wait_for_text(page: Page, pattern: str, timeout_s: int, caption: str) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if page.get_by_text(pattern).count():
            return True
        keep_caption(page, caption)
        page.wait_for_timeout(3000)
    return False


def smooth_scroll(page: Page, px: int, steps: int = 20) -> None:
    page.mouse.move(650, 450)  # over the feed; the wheel scrolls whatever is under the cursor
    for _ in range(steps):
        page.mouse.wheel(0, px / steps)
        page.wait_for_timeout(120)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=SIZE, record_video_dir=str(OUT), record_video_size=SIZE)
        page = ctx.new_page()
        global T0
        T0 = time.monotonic()  # the webm starts with the page
        page.goto(UI)
        page.wait_for_selector("text=Analyze this match", timeout=60000)
        page.wait_for_timeout(1500)

        # 1. the problem + the pro-match flow
        scene(page, "intro", "ShuttleCast: an AI agent that writes a coaching note for every rally of a badminton match")
        page.wait_for_timeout(5000)
        scene(page, "submit", "Submit a match → API Gateway → Lambda → SQS queue → Fargate worker")
        page.mouse.move(150, 440)
        page.wait_for_timeout(1200)
        page.get_by_role("button", name="Analyze this match").click()
        page.wait_for_timeout(6000)

        cap = "The worker runs a ReAct agent per rally; finished rallies stream in as they complete"
        scene(page, "wait_match", cap, speed=8)
        wait_for_text(page, "Rally 1", 300, cap + "   ⏩ 8× speed")
        cap2 = "Each card is one rally: the agent decides skip / brief / deep, then writes what happened and what to try"
        scene(page, "stream_match", cap2, speed=8)
        wait_for_text(page, "Match so far", 900, cap2 + "   ⏩ 8× speed")
        page.wait_for_timeout(2000)

        # 2. read the results
        scene(page, "results", "Done: 10 rallies. Deep analysis where a tactical error decided the rally, brief notes elsewhere")
        page.wait_for_timeout(5000)
        smooth_scroll(page, 700)
        page.wait_for_timeout(3500)
        scene(page, "watch", "▶ jumps the broadcast to that rally's first stroke")
        watch = page.get_by_role("button", name="▶")
        if watch.count() > 2:
            watch.nth(2).click()
        page.wait_for_timeout(6000)

        # 3. grounded Q&A
        scene(page, "qa", "Ask about the match: answers are grounded in this match's stroke data and the agent's notes")
        smooth_scroll(page, 4000, steps=25)
        ask = page.get_by_role("button", name="Where did B lose points?")
        if ask.count():
            ask.first.click()
        else:
            page.get_by_placeholder("Ask about this match").fill("Where did B lose points?")
            page.keyboard.press("Enter")
        wait_for_text(page, "lost", 90, "Ask about the match: answers are grounded in this match's stroke data and the agent's notes")
        page.wait_for_timeout(2500)
        smooth_scroll(page, 4000, steps=10)
        page.wait_for_timeout(6000)

        # 4. footage nobody annotated
        scene(page, "upload", "Your own video: no annotations. Stroke models trained on ShuttleSet find the hits and shot types")
        page.get_by_text("Your video", exact=True).click()
        page.wait_for_timeout(3000)
        page.locator("input[type=file]").set_input_files(str(UPLOAD))
        page.wait_for_timeout(8000)
        page.get_by_role("button", name="Analyze", exact=True).click()
        page.wait_for_timeout(5000)

        cap3 = "Fargate worker: DINOv2 features → hit detector → shot classifier → same agent per detected rally"
        scene(page, "wait_upload", cap3, speed=12)
        wait_for_text(page, "Rally 1", 900, cap3 + "   ⏩ 12× speed")
        cap4 = "Detected rallies stream in. The models never saw this match in training (held-out test match)"
        scene(page, "stream_upload", cap4, speed=12)
        wait_for_text(page, "Match so far", 900, cap4 + "   ⏩ 12× speed")
        page.wait_for_timeout(2000)

        scene(page, "upload_results", "Coaching notes for footage nobody labelled, with the uploaded video alongside")
        page.wait_for_timeout(5000)
        smooth_scroll(page, 600)
        page.wait_for_timeout(5000)
        scene(page, "outro", "Hit detection F1 0.87 on raw broadcast · player attribution 62% → 94% · github.com/Naramkeshav59/Shuttlecast")
        page.wait_for_timeout(7000)
        scenes.append({"name": "end", "t": time.monotonic() - T0, "speed": 1})

        video = page.video.path()
        ctx.close()
        browser.close()
    (OUT / "scenes.json").write_text(json.dumps({"video": str(video), "scenes": scenes}, indent=1))
    print(video)


if __name__ == "__main__":
    main()
