#!/usr/bin/env python3
"""Pre-submission sanity check + smoke mode.

Run from repo root: `make verify` (or `python scripts/verify.py`).
For a quick smoke run before training: `python scripts/verify.py --smoke`.

Exits 0 if every required artifact is present + REFLECTION.md edited beyond the
template. Exits non-zero with a checklist of what's missing — no files written.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Windows PowerShell 5 commonly defaults to cp1252, which cannot render the
# Vietnamese/Unicode diagnostics in this lab. Make the gatekeeper report its
# checklist instead of crashing before it reaches the real failures.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TEMPLATE_MARKERS = [
    r"<Họ Tên>",
    r"<A20-K1 / A20-K2",
    r"<YYYY-MM-DD>",
    r"_Answer here\.\s*≥",
    r"_Answer here\._?\s*$",
    r"<e\.g\., Free Colab T4 16GB",
]
REQUIRED_REFLECTION_SECTIONS = [
    "## 1. Setup",
    "## 2. DPO experiment results",
    "## 3. Reward curves analysis",
    "## 4. Qualitative comparison",
    "## 5. β trade-off",
    "## 6. Personal reflection",
]


def check_file(path: Path, label: str, problems: list[str]) -> bool:
    if not path.exists():
        problems.append(f"MISSING  {label}: {path.relative_to(Path.cwd())}")
        return False
    if path.stat().st_size == 0:
        problems.append(f"EMPTY    {label}: {path.relative_to(Path.cwd())}")
        return False
    return True


def check_screenshots(folder: Path, min_count: int, problems: list[str]) -> int:
    if not folder.exists():
        problems.append("MISSING  submission/screenshots/ folder")
        return 0
    images = [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if len(images) < min_count:
        problems.append(
            f"TOO FEW  submission/screenshots/: have {len(images)}, need at least {min_count}. "
            f"See submission/screenshots/README.md for the list."
        )
    return len(images)


def check_reflection_edited(path: Path, problems: list[str]) -> bool:
    if not path.exists():
        problems.append("MISSING  submission/REFLECTION.md")
        return False
    text = path.read_text(encoding="utf-8")
    leftover = []
    for pattern in TEMPLATE_MARKERS:
        flags = re.MULTILINE if pattern.startswith("^") else 0
        if re.search(pattern, text, flags):
            leftover.append(pattern)
    if len(leftover) >= 3:
        problems.append(
            f"UNEDITED submission/REFLECTION.md still has {len(leftover)} template placeholders. "
            f"Fill in your own numbers and answers."
        )
        return False
    missing = [heading for heading in REQUIRED_REFLECTION_SECTIONS if heading not in text]
    if missing:
        problems.append(f"INCOMPLETE submission/REFLECTION.md missing sections: {', '.join(missing)}")
        return False
    # The required analyses must be substantial rather than just a heading and
    # a screenshot link. Count the text until the next H2 heading.
    for heading in ("## 3. Reward curves analysis", "## 6. Personal reflection"):
        body = text.split(heading, 1)[1].split("\n## ", 1)[0]
        words = re.findall(r"\b[\wÀ-ỹ][\wÀ-ỹ'-]*\b", body)
        if len(words) < 150:
            problems.append(
                f"TOO SHORT submission/REFLECTION.md {heading}: {len(words)} words, need at least 150"
            )
            return False
    return True


def check_adapter_config(path: Path, label: str, problems: list[str], *, sft: bool = False) -> bool:
    if not check_file(path, label, problems):
        return False
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"CORRUPT  {label}: {exc}")
        return False
    if sft and (config.get("lora_alpha") != 32 or config.get("r") != 16):
        problems.append(
            f"WRONG    {label}: expected lora_alpha=32 and r=16, got "
            f"lora_alpha={config.get('lora_alpha')}, r={config.get('r')}"
        )
        return False
    return True


def check_dpo_metrics(repo: Path, problems: list[str]) -> bool:
    metrics_path = repo / "adapters" / "dpo" / "dpo_metrics.json"
    if not metrics_path.exists():
        problems.append("MISSING  adapters/dpo/dpo_metrics.json (NB3 didn't complete)")
        return False
    try:
        m = json.loads(metrics_path.read_text())
    except Exception as exc:
        problems.append(f"CORRUPT  adapters/dpo/dpo_metrics.json — {exc}")
        return False
    gap = m.get("end_reward_gap")
    if gap is None:
        problems.append("WARN     adapters/dpo/dpo_metrics.json has no end_reward_gap (TRL log columns missing?)")
        return True
    if gap <= 0:
        problems.append(
            f"FAILED   end_reward_gap = {gap:+.3f} (≤ 0). DPO did not separate chosen from rejected."
        )
        return False
    start_gap = m.get("start_reward_gap")
    if start_gap is None:
        problems.append("MISSING  start_reward_gap in dpo_metrics.json; rerun updated NB3")
        return False
    if gap <= start_gap:
        problems.append(
            f"FAILED   reward gap did not increase: start={start_gap:+.3f}, end={gap:+.3f}. "
            "Try beta=0.05 or lr=1e-6, then rerun NB3."
        )
        return False
    return True


def check_gguf(repo: Path, problems: list[str]) -> bool:
    gguf_dir = repo / "gguf"
    if not gguf_dir.exists():
        problems.append("MISSING  gguf/ directory (NB5 didn't run)")
        return False
    files = list(gguf_dir.glob("*.gguf"))
    if not files:
        problems.append("MISSING  gguf/*.gguf — NB5 quantization step didn't write a file")
        return False
    big = [p for p in files if p.stat().st_size > 5 * 1024**3]
    if big:
        problems.append(
            f"OVERSIZED  GGUF files > 5 GB: {[p.name for p in big]}. "
            f"Q4_K_M should be ≤ 5 GB even on 7B."
        )
    return True


def smoke_check(repo: Path) -> int:
    """Light-weight pre-training check: imports work, GPU visible, deck files present."""
    print("==> Smoke check (imports + GPU + deck files)\n")
    problems: list[str] = []

    # Imports
    try:
        import torch  # noqa: WPS433
        print(f"  ✓ torch              {torch.__version__}")
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            print(f"  ✓ CUDA               {dev.name} ({dev.total_memory / 1e9:.1f} GB)")
        else:
            problems.append("torch.cuda.is_available() == False -- DPO needs a CUDA/ROCm GPU. Use the Colab T4 path (see HARDWARE-GUIDE.md); this local smoke gate cannot pass on CPU/Mac.")
    except ImportError as exc:
        problems.append(f"torch import failed: {exc}")

    for mod in ["unsloth", "trl", "peft", "bitsandbytes", "datasets", "matplotlib"]:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except (ImportError, NotImplementedError, RuntimeError) as exc:
            # unsloth raises NotImplementedError (not ImportError) when no GPU is present.
            problems.append(f"{mod} import failed: {exc}")

    # Deck source (sibling file)
    deck = repo.parent / "day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex"
    if deck.exists():
        print(f"  ✓ deck source        {deck.name}")
    else:
        print(f"  ⚠ deck source not found at {deck} — fine if you cloned standalone")

    # Notebook source files
    nb_dir = repo / "notebooks"
    expected_nbs = [
        "01_sft_mini.py", "02_preference_data.py", "03_dpo_train.py",
        "04_compare_and_eval.py", "05_merge_deploy_gguf.py", "06_benchmark.py",
    ]
    for nb in expected_nbs:
        if (nb_dir / nb).exists():
            print(f"  ✓ {nb}")
        else:
            problems.append(f"missing notebook {nb_dir / nb}")

    # NB6 benchmark dependency check
    try:
        import lm_eval  # noqa: F401
        print(f"  ✓ lm_eval (NB6 benchmark suite)")
    except ImportError:
        print("  ⚠ lm_eval missing — optional NB6 benchmark is unavailable")

    print()
    if problems:
        print("✗ Smoke check FAILED:\n")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("✓ Smoke check passed. You can now run `make pipeline` (or open a notebook).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run pre-training smoke check (imports + GPU) instead of submission gatekeeper",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent

    if args.smoke:
        return smoke_check(repo)

    problems: list[str] = []
    print(f"==> Verifying submission readiness at {repo}\n")

    # Notebook source files
    for nb in ["01_sft_mini.py", "02_preference_data.py", "03_dpo_train.py",
               "04_compare_and_eval.py", "05_merge_deploy_gguf.py"]:
        check_file(repo / "notebooks" / nb, f"notebook {nb}", problems)

    # Adapter outputs
    check_adapter_config(repo / "adapters" / "sft-mini" / "adapter_config.json",
                         "SFT-mini adapter config (NB1 output)", problems, sft=True)
    check_adapter_config(repo / "adapters" / "dpo" / "adapter_config.json",
                         "DPO adapter config (NB3 output)", problems)
    dpo_provenance = repo / "adapters" / "dpo" / "dpo_training_config.json"
    if check_file(dpo_provenance, "DPO training provenance (NB3 output)", problems):
        try:
            provenance = json.loads(dpo_provenance.read_text(encoding="utf-8"))
            if provenance.get("method") != "DPO":
                problems.append("WRONG    adapters/dpo/dpo_training_config.json method must be DPO")
        except json.JSONDecodeError as exc:
            problems.append(f"CORRUPT  adapters/dpo/dpo_training_config.json — {exc}")

    # DPO metrics
    check_dpo_metrics(repo, problems)

    # Preference data
    check_file(repo / "data" / "pref" / "train.parquet",
               "preference data parquet (NB2 output)", problems)

    # Eval outputs
    check_file(repo / "data" / "eval" / "side_by_side.jsonl",
               "side-by-side eval (NB4 output)", problems)
    check_file(repo / "data" / "eval" / "judge_results.json",
               "judge results (NB4 output)", problems)

    # OPTIONAL (bonus) — NB5 GGUF + NB6 benchmark: report, do NOT fail core
    optional = []
    if not list((repo / "gguf").glob("*.gguf")):
        optional.append("NB5 GGUF (gguf/*.gguf) not done")
    if not (repo / "data" / "eval" / "benchmark_results.json").exists():
        optional.append("NB6 benchmark (data/eval/benchmark_results.json) not done")

    # Submission artifacts (core)
    check_reflection_edited(repo / "submission" / "REFLECTION.md", problems)
    n_shots = check_screenshots(repo / "submission" / "screenshots", min_count=6, problems=problems)
    if n_shots:
        print(f"  ✓ submission/screenshots/ has {n_shots} image(s)")

    if optional:
        print("\nⓘ Optional (bonus) not done — fine for a core pass:")
        for line in optional:
            print(f"  - {line}")

    print()
    if not problems:
        print("✓ Core checks passed. Push your repo (public!) and paste the URL into LMS.")
        return 0

    print("✗ Submission not ready yet:\n")
    for line in problems:
        print(f"  - {line}")
    print(
        "\nFix the items above and rerun `make verify`. See rubric.md for full grading details."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
