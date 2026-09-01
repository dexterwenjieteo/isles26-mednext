# PowerShell helper to run the ISLES26 Docker build/test/save on Windows.
#
# Usage (from PowerShell, in the isles26_submission folder):
#   .\do_build.ps1        # build the Docker image
#   .\do_test.ps1         # run the end-to-end /health + /invoke test
#   .\do_save.ps1         # export image + model tarballs for upload
#
# Requires:
#   - Docker Desktop running (verify with: docker --version)
#   - Git for Windows (provides bash) OR WSL
#
# These just invoke the corresponding bash scripts via Git Bash / WSL.

param(
    [ValidateSet("build", "test", "save")]
    [string]$Action = "build"
)

$ErrorActionPreference = "Stop"

# Locate bash (Git Bash or WSL).
$bash = $null
foreach ($candidate in @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files (x86)\Git\bin\bash.exe"
)) {
    if (Test-Path $candidate) { $bash = $candidate; break }
}
if (-not $bash) {
    # Try WSL
    if (Get-Command wsl -ErrorAction SilentlyContinue) {
        Write-Host "Using WSL bash..."
        switch ($Action) {
            "build" { wsl -e bash ./do_build.sh }
            "test"  { wsl -e bash ./do_test_run.sh }
            "save"  { wsl -e bash ./do_save.sh }
        }
        exit
    }
    Write-Error "bash not found. Install Git for Windows (provides Git Bash) or WSL."
    exit 1
}

Write-Host "Using bash: $bash"
switch ($Action) {
    "build" { & $bash ./do_build.sh }
    "test"  { & $bash ./do_test_run.sh }
    "save"  { & $bash ./do_save.sh }
}
