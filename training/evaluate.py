"""Evaluate the untuned baseline vs the fine-tuned adapter on the held-out match.

Run it twice on the same eval file -- once without --adapter (baseline),
once with -- so the before/after numbers come from identical inputs.

    python training/evaluate.py --out eval_baseline.json
    python training/evaluate.py --adapter checkpoints/qwen2vl-shuttlecast-qlora/adapter --out eval_finetuned.json

Reports: format compliance, tactical accuracy (blames the rally's loser),
depth / suggestion-type agreement with the teacher, and suggestion
similarity to the teacher. The teacher's own tactical accuracy is reported
as the ceiling a distilled student can be expected to approach.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.models import AnalysisResult, RallyReport  # noqa: E402
from training.finetune import load_rows  # noqa: E402
from worker.eval.metrics import suggestion_similarity, tactical_accuracy  # noqa: E402
from worker.pipeline.inference import FineTunedAnalyzer, parse_output, resolve  # noqa: E402


def to_report(row: dict, parsed: dict) -> RallyReport:
    analysis = AnalysisResult(match_id=row["match_id"], set_num=row["set_num"], rally_id=row["rally_id"], **parsed)
    return RallyReport(analysis=analysis, start_time_sec=0.0, rally_winner=row["rally_winner"],
                       n_strokes=row["stroke_summary"]["rally_length"], score_a=0, score_b=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for the untuned baseline")
    parser.add_argument("--eval-file", default=str(REPO_ROOT / "data/training/eval.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--base-model", default=None, help="override the base model (smoke tests)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.eval_file))[: args.limit]
    kwargs = {"base_model": args.base_model} if args.base_model else {}
    analyzer = FineTunedAnalyzer(adapter_path=args.adapter, load_in_4bit=not args.no_quant, **kwargs)

    student, teacher, records = [], [], []
    gen_suggestions, ref_suggestions = [], []
    depth_agree = type_agree = type_scored = 0
    for i, row in enumerate(rows, 1):
        frames = [resolve(p, REPO_ROOT) for p in row["frames"]]
        raw = analyzer.generate(frames, row["stroke_summary"], row["rally_winner"])
        parsed = parse_output(raw)
        ref = parse_output(row["target"])
        records.append({"id": row["id"], "raw": raw, "parsed": parsed, "teacher": ref})
        teacher.append(to_report(row, ref))
        if parsed is None:
            continue
        student.append(to_report(row, parsed))
        depth_agree += parsed["depth"] == ref["depth"]
        if ref["depth"] != "skip" and parsed["depth"] != "skip":
            type_scored += 1
            type_agree += parsed["suggestion_type"] == ref["suggestion_type"]
            gen_suggestions.append(parsed["suggestion_text"])
            ref_suggestions.append(ref["suggestion_text"])
        print(f"[{i}/{len(rows)}] {row['id']}: {parsed['depth'] if parsed else 'UNPARSEABLE'}", file=sys.stderr)

    n_parsed = len(student)
    report = {
        "model": args.adapter or "baseline (untuned)",
        "n": len(rows),
        "format_compliance": n_parsed / len(rows) if rows else None,
        "tactical_accuracy": tactical_accuracy(student),
        "teacher_tactical_accuracy": tactical_accuracy(teacher),
        "depth_agreement_with_teacher": depth_agree / n_parsed if n_parsed else None,
        "type_agreement_with_teacher": type_agree / type_scored if type_scored else None,
        "suggestion_similarity_to_teacher": suggestion_similarity(gen_suggestions, ref_suggestions),
        "records": records,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
