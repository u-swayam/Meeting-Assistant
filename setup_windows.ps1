# setup_windows.ps1
# Reproducible Windows setup for the Meeting Assistant environment.
#
# This script:
#   1. Uses Python 3.12
#   2. Creates .venv if needed
#   3. Installs PyTorch 2.8.0 + CUDA 12.8
#   4. Installs the pinned Python dependencies
#   5. Installs FFmpeg 7.1 shared through winget
#   6. Makes the FFmpeg 7.1 DLLs visible to TorchCodec
#   7. Verifies torch/CUDA, TorchCodec, and pyannote
#   8. Creates .env from .env.example (if missing) and runs the offline tests
#
# API keys are intentionally NOT handled here.

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "`n=== Meeting Assistant Windows Setup ===`n" -ForegroundColor Cyan

# ---- Python 3.12 ------------------------------------------------------------
Write-Host "[1/7] Checking Python 3.12..." -ForegroundColor Yellow

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py.exe) was not found. Install Python 3.12 first."
}

$py312 = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $py312) {
    throw "Python 3.12 was not found. Install Python 3.12 and rerun this script."
}

Write-Host "Using: $py312"

# ---- Virtual environment ----------------------------------------------------
Write-Host "`n[2/7] Creating/checking .venv..." -ForegroundColor Yellow

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv."
    }
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

# ---- PyTorch -----------------------------------------------------------------
Write-Host "`n[3/7] Installing PyTorch 2.8.0 + CUDA 12.8..." -ForegroundColor Yellow

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

& $VenvPython -m pip install `
    "torch==2.8.0" `
    "torchvision==0.23.0" `
    "torchaudio==2.8.0" `
    --index-url "https://download.pytorch.org/whl/cu128"

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installation failed."
}

# ---- Python dependencies -----------------------------------------------------
Write-Host "`n[4/7] Installing project dependencies..." -ForegroundColor Yellow

& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "requirements.txt installation failed."
}

# ---- FFmpeg shared 7.1 ------------------------------------------------------
Write-Host "`n[5/7] Installing/checking FFmpeg 7.1 shared..." -ForegroundColor Yellow

$ffmpegPackage = Get-ChildItem `
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
    -Directory `
    -Filter "BtbN.FFmpeg.GPL.Shared.7.1_*" `
    -ErrorAction SilentlyContinue |
    Select-Object -First 1

if (-not $ffmpegPackage) {
    Write-Host "FFmpeg 7.1 shared not found. Installing with winget..."
    & winget install --id BtbN.FFmpeg.GPL.Shared.7.1 --exact --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg 7.1 shared installation failed."
    }

    $ffmpegPackage = Get-ChildItem `
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
        -Directory `
        -Filter "BtbN.FFmpeg.GPL.Shared.7.1_*" `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

if (-not $ffmpegPackage) {
    throw "Could not locate the installed FFmpeg 7.1 shared package."
}

$ffmpegBin = Get-ChildItem $ffmpegPackage.FullName -Directory -Filter "ffmpeg-*\bin" -ErrorAction SilentlyContinue |
    Select-Object -First 1

if (-not $ffmpegBin) {
    throw "Could not locate the FFmpeg 7.1 shared bin directory."
}

$ffmpegBinPath = $ffmpegBin.FullName
Write-Host "FFmpeg 7.1 bin: $ffmpegBinPath"

# Make this process use FFmpeg 7.1 first.
$env:PATH = "$ffmpegBinPath;$env:PATH"

# ---- TorchCodec Windows DLL workaround --------------------------------------
Write-Host "`n[6/7] Installing TorchCodec FFmpeg DLL workaround..." -ForegroundColor Yellow

$torchcodecDir = Join-Path $ProjectRoot ".venv\Lib\site-packages\torchcodec"

if (-not (Test-Path $torchcodecDir)) {
    throw "TorchCodec package directory was not found: $torchcodecDir"
}

$dlls = Get-ChildItem $ffmpegBinPath -Filter "*.dll" -File

if (-not $dlls) {
    throw "No FFmpeg DLLs were found in $ffmpegBinPath"
}

Copy-Item $dlls.FullName $torchcodecDir -Force

Write-Host "Copied $($dlls.Count) FFmpeg DLLs into TorchCodec."

# ---- Verification ------------------------------------------------------------
Write-Host "`n[7/7] Verifying installation..." -ForegroundColor Yellow

& $VenvPython -c @"
import torch
import torchcodec
import pyannote.audio

print("Python:       ", __import__("sys").version.split()[0])
print("PyTorch:      ", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:          ", torch.cuda.get_device_name(0))
print("TorchCodec:   ", torchcodec.__version__)
print("pyannote:     ", pyannote.audio.__version__)
"@

if ($LASTEXITCODE -ne 0) {
    throw "Verification failed."
}

# ---- .env + offline tests ----------------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example -> open it and fill in your API keys." -ForegroundColor Yellow
} else {
    Write-Host ".env already exists - left untouched."
}
New-Item -ItemType Directory -Force -Path "outputs", "uploads" | Out-Null

Write-Host "`nRunning offline tests (no API keys needed)..." -ForegroundColor Yellow
& $VenvPython -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "Offline tests failed."
}

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Next:"
Write-Host "  1. Edit .env (DEEPGRAM_API_KEY, HF_TOKEN, DEEPSEEK_API_KEY)"
Write-Host "  2. .venv\Scripts\Activate.ps1"
Write-Host "  3. python -m app.server     (then open http://127.0.0.1:8000)"
Write-Host ""
Write-Host "Do NOT put API keys in this script."
