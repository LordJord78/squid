# Downloads the engine Squid runs: an official llama.cpp release from github.com/ggml-org/llama.cpp,
# pinned to one build and checked against its SHA-256 before anything is unpacked.
# The CUDA 13 build suits the RTX 5080 (driver 617, compute capability 12.0).
#
#   pwsh -File scripts\get-engine.ps1
$ErrorActionPreference = "Stop"
$tag = "b11580"
$files = [ordered]@{
    "llama-$tag-bin-win-cuda-13.4-x64.zip"  = "1a0c7d2b3ccc94a8bbffb9d8119f892f3579138669f99a07a2252ce0d45b14e8" # the programs
    "cudart-llama-bin-win-cuda-13.4-x64.zip" = "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668" # NVIDIA's CUDA runtime DLLs
}
$runtime = Join-Path $PSScriptRoot "..\runtime"
$engine = Join-Path $runtime "llama.cpp"
$downloads = Join-Path $runtime "downloads"
New-Item -ItemType Directory -Force $engine, $downloads | Out-Null

foreach ($name in $files.Keys) {
    $zip = Join-Path $downloads $name
    if (-not (Test-Path $zip)) {
        Write-Host "downloading $name"
        curl.exe --fail --location --silent --show-error --output $zip "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$name"
        if ($LASTEXITCODE -ne 0) { Remove-Item $zip -ErrorAction SilentlyContinue; throw "download failed: $name" }
    }
    # A file that doesn't match the published checksum is deleted, never unpacked.
    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $files[$name]) { Remove-Item $zip; throw "checksum mismatch for $name (got $hash)" }
    Expand-Archive $zip -DestinationPath $engine -Force
}
Set-Content (Join-Path $engine "VERSION") $tag
Write-Host "llama.cpp $tag ready in $engine"
