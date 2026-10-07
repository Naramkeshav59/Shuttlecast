"""QLoRA fine-tuning of Qwen2-VL-2B on the ShuttleCast dataset.

Runs on a CUDA box (RunPod -- see training/runpod.sh). Loss is computed on
the assistant answer only: training on the prompt tokens too would spend
capacity learning to reproduce stroke JSON instead of the analysis.

    python training/finetune.py --config training/configs/qlora_config.yaml
    python training/finetune.py --config ... --smoke   # CPU, tiny model, 2 steps
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from worker.pipeline.inference import (  # noqa: E402
    MAX_PIXELS, MIN_PIXELS, build_messages, load_images, resolve, select_frames,
)

# Random-weight model with Qwen2-VL's architecture, for exercising the
# whole pipeline on CPU before spending GPU credits.
SMOKE_MODEL = "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration"
IGNORE = -100


def load_rows(path: Path) -> list[dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row  # dedupe: prepare_dataset is resumable/appending
    return list(rows.values())


def row_messages(row: dict) -> list[dict]:
    frames = [resolve(p, REPO_ROOT) for p in select_frames(row["frames"])]
    return build_messages(frames, row["stroke_summary"], row["rally_winner"], target=row["target"])


class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.assistant_header = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def _answer_start(self, ids: list[int]) -> int:
        """Index just past the last '<|im_start|>assistant\\n'."""
        h = self.assistant_header
        for i in range(len(ids) - len(h), -1, -1):
            if ids[i:i + len(h)] == h:
                return i + len(h)
        raise ValueError("assistant header not found -- chat template changed?")

    def __call__(self, rows: list[dict]) -> dict:
        texts, images = [], []
        for row in rows:
            messages = row_messages(row)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False))
            images.extend(load_images(messages))
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for i, ids in enumerate(batch["input_ids"].tolist()):
            labels[i, : self._answer_start(ids)] = IGNORE
        labels[batch["attention_mask"] == 0] = IGNORE
        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "training/configs/qlora_config.yaml"))
    parser.add_argument("--smoke", action="store_true", help="CPU, tiny random model, 2 steps, no quantization")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg = cfg["training"]

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments,
    )

    base = SMOKE_MODEL if args.smoke else cfg["base_model"]
    use_cuda = torch.cuda.is_available() and not args.smoke
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)

    quant = None
    if cfg["quantization"]["load_in_4bit"] and not args.smoke:
        q = cfg["quantization"]
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"], bnb_4bit_compute_dtype=dtype,
        )

    processor = AutoProcessor.from_pretrained(base, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base, quantization_config=quant, dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tcfg["gradient_checkpointing"])
    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    train_rows = load_rows(REPO_ROOT / cfg["train_file"])
    eval_rows = load_rows(REPO_ROOT / cfg["eval_file"])
    if args.smoke:
        train_rows, eval_rows = train_rows[:2], eval_rows[:1]
    print(f"train={len(train_rows)} eval={len(eval_rows)}")

    out_dir = REPO_ROOT / cfg["output_dir"] / ("smoke" if args.smoke else "")
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tcfg["epochs"],
        max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        # transformers 5 folded warmup_ratio into warmup_steps (a float < 1 is a ratio)
        warmup_steps=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        max_grad_norm=tcfg["max_grad_norm"],
        logging_steps=1 if args.smoke else tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=1 if args.smoke else tcfg["eval_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        gradient_checkpointing=tcfg["gradient_checkpointing"] and use_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # the collator needs the raw rows; Trainer would otherwise drop "unused" columns
        remove_unused_columns=False,
        report_to="none",
        seed=tcfg["seed"],
        use_cpu=not use_cuda,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_rows, eval_dataset=eval_rows,
        data_collator=Collator(processor),
    )
    trainer.train()
    model.save_pretrained(out_dir / "adapter")
    processor.save_pretrained(out_dir / "adapter")
    print(f"adapter saved to {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
