<#
Sets up the Day 22 core pipeline on Windows + an NVIDIA Ampere GPU.

The RTX 3060 uses compute capability 8.6. This script deliberately installs
the CUDA 12.4 PyTorch wheel before the rest of the packages, then pins the
matching Windows xFormers wheel. A plain `pip install -r requirements.txt`
may otherwise replace the CUDA wheel with a newer CPU-only PyTorch wheel.

NB5 (llama-cpp-python) and NB6 (lm-eval) are optional and intentionally not
installed here: neither is required by the submission gatekeeper.
#>
[CmdletBinding()]
param(
    [string]$VenvPath = ".venv-gpu",
    [string]$PythonExe = "py"
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot "$VenvPath\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($PythonExe -eq "py") {
        & py -3.11 -m venv $VenvPath
    }
    else {
        & $PythonExe -m venv $VenvPath
    }
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Could not create $VenvPath with Python 3.11 x64. Install Python 3.11, then rerun."
}

& $venvPython -m pip install --upgrade pip setuptools wheel

# CUDA runtime is bundled in the wheel; a local CUDA Toolkit is not required.
& $venvPython -m pip install --no-cache-dir `
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 `
    --index-url https://download.pytorch.org/whl/cu124

# xFormers must match Torch 2.5.1 + CUDA 12.4. Letting pip choose its latest
# PyPI build is what causes Windows to attempt a Torch 2.10 replacement.
& $venvPython -m pip install --no-cache-dir --no-deps `
    "https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp311-cp311-win_amd64.whl"

# xFormers 0.0.29 was built against Triton 3.1. The newest triton-windows
# changes its JIT API and makes xFormers fall back to slow PyTorch attention.
& $venvPython -m pip install --no-cache-dir --force-reinstall triton-windows==3.1.0.post17

& $venvPython -m pip install --no-cache-dir `
    unsloth==2025.10.12 unsloth_zoo==2025.10.13 `
    trl==0.18.2 "transformers>=4.51.3,<5.0" "peft>=0.13,<1.0" `
    "accelerate>=1.1,<2.0" "datasets>=3.4,<4.0" "bitsandbytes>=0.45.5,<1.0" `
    "matplotlib>=3.9,<4.0" "pandas>=2.2,<3.0" "pyarrow>=17,<22" `
    "jupyterlab>=4.3,<5.0" "jupytext>=1.16,<2.0" "pytest>=8.3,<9.0"

# The released torchao >= 0.13 requires Torch's int1 dtype, introduced after
# Torch 2.5.1. It is optional for this QLoRA/DPO path, so remove it before
# Transformers probes optional quantizers during the Unsloth import.
& $venvPython -m pip uninstall -y torchao

# See scripts/patch_unsloth_windows.py for why this only excludes GRPO.
& $venvPython scripts\patch_unsloth_windows.py
& $venvPython -m ipykernel install --user --name day22-dpo-gpu --display-name "Python (Day 22 GPU)"

& $venvPython -c "import unsloth, torch; from trl import DPOConfig, SFTConfig; assert torch.cuda.is_available(); print(f'GPU ready: {torch.cuda.get_device_name(0)} | torch={torch.__version__} | unsloth={unsloth.__version__}')"

Write-Host ""
Write-Host "Setup complete. In VS Code choose interpreter: $venvPython"
Write-Host "Then execute notebooks 01 -> 04, or use the commands in README.md."
