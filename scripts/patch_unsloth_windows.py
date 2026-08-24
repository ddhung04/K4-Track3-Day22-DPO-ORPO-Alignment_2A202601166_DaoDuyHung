"""Apply the narrow Unsloth/TRL compatibility patch needed on Windows.

Unsloth 2025.10.12 eagerly patches every TRL trainer at import time. Its GRPO
patch expects a newer GRPOTrainer than the TRL 0.18.2 version used by this lab,
so importing Unsloth fails before the SFT/DPO trainers are reached. The lab
does not use GRPO. Excluding only that trainer preserves Unsloth's SFT and DPO
patches while keeping the course's TRL < 0.20 constraint.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        import unsloth  # noqa: F401
    except Exception:
        # Importing triggers the exact upstream error we are fixing; the module
        # location is still deterministic inside this virtual environment.
        import site

        candidates = [Path(root) / "unsloth" / "models" / "rl.py" for root in site.getsitepackages()]
    else:
        candidates = [Path(unsloth.__file__).parent / "models" / "rl.py"]

    target = next((path for path in candidates if path.is_file()), None)
    if target is None:
        raise SystemExit("Unsloth is not installed in this Python environment.")

    original = 'all_trainers = [x for x in all_trainers if x.islower() and x.endswith("_trainer")]'
    replacement = '''all_trainers = [
        x for x in all_trainers
        if x.islower() and x.endswith("_trainer") and x != "grpo_trainer"
    ]'''
    text = target.read_text(encoding="utf-8")

    if replacement in text:
        print(f"Unsloth Windows patch already present: {target}")
        return
    if original not in text:
        raise SystemExit(f"Expected Unsloth 2025.10.12 patch point was not found: {target}")

    target.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    print(f"Patched Unsloth GRPO import on Windows: {target}")


if __name__ == "__main__":
    main()
