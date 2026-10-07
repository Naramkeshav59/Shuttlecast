"""Fine-tuned Qwen2-VL-2B inference wrapper.

build_messages() is the ONE prompt format used by training/prepare_dataset.py,
training/finetune.py and inference here -- a prompt that drifts between
training and serving quietly erases a fine-tune's gains.

torch/transformers are imported lazily: the Fargate worker image doesn't
ship them (no GPU there), only the RunPod training/eval environment does.
"""
import json
import re
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MAX_FRAMES = 4
# Qwen2-VL spends (H/28)*(W/28) tokens per image; capping pixels keeps four
# 720p frames at ~1k image tokens total instead of ~5k.
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

SYSTEM_PROMPT = (
    "You are a badminton tactical analyst. Given broadcast frames from the decisive "
    "strokes of one rally and that rally's stroke data, identify the tactical error "
    "that decided the rally and suggest a correction."
)
DEPTHS = ("deep", "surface", "skip")
SUGGESTION_TYPES = ("positioning", "shot_selection", "pattern_exploitation")


def select_frames(frame_paths: list, max_frames: int = MAX_FRAMES) -> list:
    """The last strokes are where a rally is decided."""
    return list(frame_paths)[-max_frames:]


def _compact(summary: dict) -> dict:
    """Round coordinates to ints -- floats cost tokens and carry no signal."""
    def pts(seq):
        return [[None if v is None else round(v) for v in p] for p in seq]

    return {
        "shots": summary["shot_sequence"],
        "hitters": summary["players"],
        "landing_xy": pts(summary["landing_zones"]),
        "hitter_xy": pts(summary["player_positions"]),
    }


def user_text(stroke_summary: dict, rally_winner: str | None) -> str:
    return (
        f"Stroke data, oldest first: {json.dumps(_compact(stroke_summary))}\n"
        f"Rally won by player {rally_winner}.\n"
        "Answer in exactly this format:\n"
        "DEPTH: deep|surface|skip\n"
        "TYPE: positioning|shot_selection|pattern_exploitation|none\n"
        "PATTERN: <one sentence naming the player who erred>\n"
        "SUGGESTION: <1-2 sentences>"
    )


def build_messages(frame_paths: list, stroke_summary: dict, rally_winner: str | None,
                   target: str | None = None) -> list[dict]:
    content = [{"type": "image", "image": str(p)} for p in frame_paths]
    content.append({"type": "text", "text": user_text(stroke_summary, rally_winner)})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def format_target(depth: str, suggestion_type: str | None, pattern: str | None, suggestion: str | None) -> str:
    return (
        f"DEPTH: {depth}\nTYPE: {suggestion_type or 'none'}\n"
        f"PATTERN: {pattern or 'none'}\nSUGGESTION: {suggestion or 'none'}"
    )


_FIELD = re.compile(r"^(DEPTH|TYPE|PATTERN|SUGGESTION):\s*(.+?)\s*$", re.MULTILINE)


def parse_output(text: str) -> dict | None:
    """Model output -> AnalysisResult fields, or None if it isn't in format."""
    fields = {k: v for k, v in _FIELD.findall(text or "")}
    depth = fields.get("DEPTH", "").lower()
    if depth not in DEPTHS:
        return None
    if depth == "skip":
        return {"depth": "skip", "suggestion_type": None, "tactical_pattern": None, "suggestion_text": None}
    stype = fields.get("TYPE", "").lower()
    if stype not in SUGGESTION_TYPES or "PATTERN" not in fields or "SUGGESTION" not in fields:
        return None
    return {
        "depth": depth, "suggestion_type": stype,
        "tactical_pattern": fields["PATTERN"], "suggestion_text": fields["SUGGESTION"],
    }


def load_images(messages: list[dict]) -> list:
    from PIL import Image

    return [
        Image.open(part["image"]).convert("RGB")
        for msg in messages for part in msg["content"] if part["type"] == "image"
    ]


class FineTunedAnalyzer:
    """Qwen2-VL-2B (+ optional LoRA adapter). adapter_path=None gives the
    untuned baseline, which is what evaluate.py compares against."""

    def __init__(self, adapter_path: str | None = None, base_model: str = BASE_MODEL,
                 load_in_4bit: bool = True):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model, quantization_config=quant, dtype=torch.bfloat16, device_map="auto",
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            base_model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )

    def generate(self, frame_paths: list, stroke_summary: dict, rally_winner: str | None,
                 max_new_tokens: int = 200) -> str:
        import torch

        messages = build_messages(select_frames(frame_paths), stroke_summary, rally_winner)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=load_images(messages), return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0]

    def analyze(self, frame_paths: list, stroke_summary: dict, rally_winner: str | None) -> dict | None:
        return parse_output(self.generate(frame_paths, stroke_summary, rally_winner))


def resolve(path: str | Path, root: Path) -> Path:
    """Dataset rows store repo-relative frame paths so they survive the trip to RunPod."""
    p = Path(path)
    return p if p.is_absolute() else root / p
