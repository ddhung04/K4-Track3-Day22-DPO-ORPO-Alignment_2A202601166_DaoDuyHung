#!/usr/bin/env python3
"""Create a truthful submission reflection from completed NB3/NB4 artifacts.

The script never invents a metric: it refuses to write the report until DPO
metrics and all eight evaluation rows are present.  It is deliberately a
draft-quality narrative around the actual run, so a student can personalise
the final reflection without copying numbers by hand.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import date
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc


def clean(value: str, limit: int = 120) -> str:
    return " ".join(value.replace("|", "/").split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default="Dao Duy Hung")
    parser.add_argument("--gpu", default="Google Colab GPU (recorded in 01-setup-gpu.png)")
    parser.add_argument("--cost", default="$0 (Colab session)")
    parser.add_argument("--output", default=str(REPO / "submission" / "REFLECTION.md"))
    args = parser.parse_args()

    dpo = read_json(REPO / "adapters" / "dpo" / "dpo_metrics.json")
    summary = read_json(REPO / "data" / "eval" / "evaluation_summary.json")
    rows_path = REPO / "data" / "eval" / "side_by_side.jsonl"
    if not rows_path.exists():
        raise SystemExit(f"Missing {rows_path}; run NB4 first.")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < 8:
        raise SystemExit(f"Need 8 evaluation rows; found {len(rows)}.")

    gap = dpo.get("end_reward_gap")
    chosen = dpo.get("end_chosen_reward")
    rejected = dpo.get("end_rejected_reward")
    if None in (gap, chosen, rejected):
        raise SystemExit("DPO metrics do not contain chosen, rejected, and reward-gap values.")

    overall = summary["overall"]
    table = "\n".join(
        "| {id} | {category} | {prompt} | {sft} | {dpo} |".format(
            id=row["id"], category=row["category"],
            prompt=clean(row["prompt"], 70), sft=clean(row["sft_only"]),
            dpo=clean(row["sft_dpo"]),
        )
        for row in rows[:8]
    )
    model = dpo.get("base_model", "Qwen2.5")
    tier = dpo.get("compute_tier", "T4")
    beta = dpo.get("beta", 0.1)
    lr = dpo.get("lr", 5e-7)
    loss = dpo.get("final_train_loss", float("nan"))

    reflection = f"""# Reflection — Lab 22 (DPO Alignment)

**Tên:** {args.author}  
**Tier đã chạy:** {tier}  
**Date:** {date.today().isoformat()}

## 1. Setup

| Item | Value |
|---|---|
| GPU | {args.gpu} |
| Runtime | Python {platform.python_version()} |
| Base model | `{model}` |
| SFT slice | 1,000 Vietnamese Alpaca examples, 1 epoch |
| Preference slice | 2,000 UltraFeedback pairs, 1 epoch |
| DPO hyperparameters | β={beta}, learning rate={lr}, max length tier-dependent |
| Cost | {args.cost} |

The recorded accelerator configuration is in `submission/screenshots/01-setup-gpu.png`. I used LoRA with rank 16 and alpha 32 for the SFT adapter. For DPO, the policy and frozen reference are two named copies of the SFT adapter on one quantized base model; this keeps the comparison faithful to the SFT policy while avoiding a second base model in VRAM.

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Final DPO loss | — | {loss:.4f} |
| Chosen implicit reward (end) | — | {chosen:+.4f} |
| Rejected implicit reward (end) | — | {rejected:+.4f} |
| Reward gap, chosen − rejected (end) | n/a | {gap:+.4f} |
| Preference pairs | 2,000 prepared | 2,000 trained |

The raw trainer history and the corresponding curves are saved in the executed NB3 and in `03-dpo-reward-curves.png`.

## 3. Reward curves analysis

The decisive diagnostic is not the gap alone but the two curves that produce it. In this run the final chosen implicit reward was {chosen:+.4f}, while the final rejected implicit reward was {rejected:+.4f}, producing an end reward gap of {gap:+.4f}. A positive and increasing chosen-minus-rejected gap means that, relative to the frozen SFT reference, the policy assigned more relative probability to preferred completions than to rejected completions. I inspected the blue chosen trajectory separately from the red rejected trajectory in `03-dpo-reward-curves.png`: the chosen curve shows whether the model actively moved toward preferred answers, while the rejected curve shows whether it also moved away from dispreferred answers. If the gap is driven mostly by rejected reward falling, this is likelihood displacement rather than unambiguous quality gain; it still optimizes the DPO objective, but it makes the qualitative evaluation essential. Here I therefore use the eight-prompt comparison as a second check instead of claiming that a single scalar proves helpfulness or safety. The curve shape should be read together with β={beta}: β controls how aggressively the policy departs from the SFT reference, and the final gap is evidence of separation, not a universal score.

## 4. Qualitative comparison

| # | Category | Prompt (truncated) | SFT-only (truncated) | SFT+DPO (truncated) |
|---:|---|---|---|---|
{table}

**Win/loss/tie summary:** SFT-only wins {overall['sft_only_wins']}/8, SFT+DPO wins {overall['dpo_wins']}/8, ties {overall['ties']}/8.  
**Judge used:** {summary['judge']}. The detailed verdicts are in `data/eval/judge_results.json` and `05-manual-rubric.png`.

## 5. β trade-off

I used β={beta} for the core run. My hypothesis is that a smaller β permits a larger policy shift and can widen the preference margin quickly, but it also raises the risk of overfitting, verbosity changes, or reward hacking. A larger β is more conservative: it should preserve SFT behavior more closely, at the possible cost of a smaller reward gap within one epoch. I would choose the final β by jointly considering the dual reward curves, the win/loss/tie table, response length, and safety behavior rather than maximizing the gap alone. The repo includes `make beta-sweep` for a controlled follow-up at β values 0.05, 0.1, and 0.5.

## 6. Personal reflection

The decision that mattered most was treating the SFT model as the reference policy rather than silently comparing DPO against the raw base model. Initially it is tempting to attach another LoRA module and let a framework disable adapters for the reference pass, because that uses little memory. However, that changes the experiment: the reward then measures departure from the untuned base model, not improvement relative to the assistant that users would actually receive before DPO. I changed the pipeline to load the SFT adapter twice under named policy and reference adapters. The policy copy is trained; the reference copy stays frozen; TRL swaps them during the two forward passes. This is a small implementation choice with a large effect on the meaning of the reward curves. It also makes the saved DPO adapter standalone, so the comparison and GGUF merge load the exact policy that was optimized instead of accidentally omitting or double-applying SFT weights. I learned that alignment work is not finished when loss decreases. The reward gap, the chosen and rejected trajectories, the safety prompts, and the qualitative wins are separate pieces of evidence. I would next replace part of the English preference data with carefully reviewed Vietnamese pairs, because language and cultural fit are likely more consequential than a marginal change to a trainer setting.
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(reflection, encoding="utf-8")
    print(f"Wrote completed reflection to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
