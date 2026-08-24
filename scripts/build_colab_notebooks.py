#!/usr/bin/env python3
"""Build the two runnable, stitched Colab notebooks from Jupytext sources.

The canonical implementation is kept in ``notebooks/*.py``.  This generator
prevents the Colab copies from drifting and deliberately creates a fresh
notebook without outputs; Colab is where the user executes it on a real GPU.
"""
from __future__ import annotations

from pathlib import Path

import jupytext
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


REPO = Path(__file__).resolve().parent.parent
CORE = [
    "01_sft_mini.py",
    "02_preference_data.py",
    "03_dpo_train.py",
    "04_compare_and_eval.py",
]
BONUS = "05_merge_deploy_gguf.py"
REPO_URL = "https://github.com/ddhung04/K4-Track3-Day22-DPO-ORPO-Alignment_2A202601166_DaoDuyHung.git"


def setup_cells(tier: str) -> list:
    return [
        new_markdown_cell(
            f"# Lab 22 — DPO Alignment ({tier} GPU tier)\n\n"
            "Run all cells after selecting a GPU runtime: "
            "**Runtime → Change runtime type → T4 GPU**. The notebook clones "
            "the current public repository into `/content/lab22`, keeps all "
            "artifacts there, and generates the required evidence files."
        ),
        new_markdown_cell("## A. Colab setup"),
        new_code_cell(
            "import os\n"
            f"os.environ['COMPUTE_TIER'] = '{tier}'\n"
            "os.environ.setdefault('STUDENT_NAME', 'Dao Duy Hung')\n"
            "os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')\n"
            "print(f\"COMPUTE_TIER={os.environ['COMPUTE_TIER']}\")"
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import subprocess, sys\n\n"
            f"REPO_URL = '{REPO_URL}'\n"
            "WORK = Path('/content/lab22')\n"
            "if (WORK / '.git').exists():\n"
            "    subprocess.run(['git', '-C', str(WORK), 'pull', '--ff-only'], check=True)\n"
            "else:\n"
            "    subprocess.run(['git', 'clone', REPO_URL, str(WORK)], check=True)\n"
            "print(f'Repository: {WORK}')\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(WORK / 'requirements.txt')], check=True)\n"
            "os.chdir(WORK / 'notebooks')\n"
            "print(f'Working directory: {Path.cwd()}')"
        ),
        new_code_cell(
            "import torch\n"
            "assert torch.cuda.is_available(), (\n"
            "    'Enable a GPU: Runtime → Change runtime type → T4 GPU, then restart and Run all.'\n"
            ")\n"
            "gpu = torch.cuda.get_device_properties(0)\n"
            "print(f'GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)')\n"
            "print(f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}')"
        ),
        new_markdown_cell("---\n\nThe next four sections are the core submission pipeline."),
    ]


def report_cells() -> list:
    return [
        new_markdown_cell("## Generate the completed submission reflection"),
        new_code_cell(
            "import subprocess, sys\n"
            "report = WORK / 'scripts' / 'generate_submission_report.py'\n"
            "subprocess.run([sys.executable, str(report), '--author', os.environ['STUDENT_NAME']], check=True)\n"
            "print((WORK / 'submission' / 'REFLECTION.md').read_text(encoding='utf-8')[:1500])"
        ),
    ]


def export_cells() -> list:
    return [
        new_markdown_cell(
            "## Export artifacts\n\n"
            "This creates one zip containing the submission screenshots, completed "
            "reflection, adapter evidence, preference data, and evaluation results. "
            "Use **File → Save a copy in Drive** as well to preserve this executed "
            "notebook, then extract the zip into the repository before `git add`."
        ),
        new_code_cell(
            "import zipfile\n"
            "# This is the same pre-submission gate as `make verify`; it must pass\n"
            "# before artifacts are exported.\n"
            "subprocess.run([sys.executable, str(WORK / 'scripts' / 'verify.py')], check=True)\n"
            "archive = Path('/content/lab22-submission-artifacts.zip')\n"
            "include = ['adapters/sft-mini', 'adapters/dpo', 'data/pref', 'data/eval', 'submission']\n"
            "with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:\n"
            "    for relative in include:\n"
            "        folder = WORK / relative\n"
            "        if folder.exists():\n"
            "            for path in folder.rglob('*'):\n"
            "                if path.is_file():\n"
            "                    zf.write(path, path.relative_to(WORK))\n"
            "print(f'Created {archive} ({archive.stat().st_size / 1e6:.1f} MB)')\n"
            "try:\n"
            "    from google.colab import files\n"
            "    files.download(str(archive))\n"
            "except ImportError:\n"
            "    print('Download is available when this notebook runs in Google Colab.')"
        ),
    ]


def build_one(tier: str, destination: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "colab": {"provenance": [], "gpuType": "T4" if tier == "T4" else "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook.cells = setup_cells(tier)
    for source in CORE:
        notebook.cells.extend(jupytext.read(REPO / "notebooks" / source).cells)
    notebook.cells.extend(report_cells())
    notebook.cells.extend(jupytext.read(REPO / "notebooks" / BONUS).cells)
    notebook.cells.extend(export_cells())
    nbformat.write(notebook, destination)
    print(f"Wrote {destination.relative_to(REPO)} ({len(notebook.cells)} cells)")


def main() -> None:
    build_one("T4", REPO / "colab" / "Lab22_DPO_T4.ipynb")
    build_one("BIGGPU", REPO / "colab" / "Lab22_DPO_BigGPU.ipynb")


if __name__ == "__main__":
    main()
