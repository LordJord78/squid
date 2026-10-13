# Records how the real LM Studio behaves, so Squid can be checked against it.
# Uses only LM Studio's public CLI (lms) and its documented local HTTP API.
# Leaves things as it found them: unloads the model and stops the server if they were off.
#
#   pwsh -File golden\capture-lmstudio.ps1 [-Model qwen/qwen3.5-9b]

param([string]$Model = "qwen/qwen3.5-9b")
$ErrorActionPreference = "Stop"
$lms = "$env:USERPROFILE\.lmstudio\bin\lms.exe"
$out = Join-Path $PSScriptRoot "lmstudio"
$base = "http://127.0.0.1:1234"
New-Item -ItemType Directory -Force $out | Out-Null

function Save($name, $text) { Set-Content -Path (Join-Path $out $name) -Value $text -Encoding utf8NoBOM }
function Get-Json($path) {
    try { (Invoke-WebRequest "$base$path" -TimeoutSec 30).Content } catch { "HTTP error: $($_.Exception.Message)" }
}
function Post($path, $body, $file) {
    $json = $body | ConvertTo-Json -Depth 10 -Compress
    # curl.exe writes the reply straight to the file, byte for byte. Capturing it in PowerShell first
    # would re-split the stream's lines and turn every "\n" into "\r\n".
    $json | curl.exe -s -N -X POST "$base$path" -H "Content-Type: application/json" --data-binary "@-" -o (Join-Path $out $file)
}

$wasRunning = (& $lms server status 2>&1 | Out-String) -notmatch "not running"
$wasLoaded  = (& $lms ps 2>&1 | Out-String) -match [regex]::Escape($Model)

& $lms server start 2>&1 | Out-Null
Save "lms-ls.txt" (& $lms ls 2>&1 | Out-String)
Save "lms-ls.json" (& $lms ls --json 2>$null | Out-String).Trim()
Save "v1-models.json" (Get-Json "/v1/models")
Save "api-v0-models.json" (Get-Json "/api/v0/models")

if (-not $wasLoaded) { & $lms load $Model --context-length 8192 -y 2>&1 | Out-Null }
Save "lms-ps.txt" (& $lms ps 2>&1 | Out-String)
Save "api-v0-models-loaded.json" (Get-Json "/api/v0/models")

$ask = @{ role = "user"; content = "Reply with exactly one word: pong" }
Post "/v1/chat/completions" @{ model = $Model; messages = @($ask); temperature = 0; max_tokens = 400 } "chat-nonstream.json"
Post "/v1/chat/completions" @{ model = $Model; messages = @($ask); temperature = 0; max_tokens = 400; stream = $true } "chat-stream.sse"
Post "/v1/chat/completions" @{ model = $Model; messages = @($ask); temperature = 0; max_tokens = 400; stream = $true; stream_options = @{ include_usage = $true } } "chat-stream-usage.sse"
Post "/api/v0/chat/completions" @{ model = $Model; messages = @($ask); temperature = 0; max_tokens = 400 } "api-v0-chat.json"

$tool = @{ type = "function"; function = @{
    name = "read_file"; description = "Read a text file from the current project"
    parameters = @{ type = "object"; properties = @{ path = @{ type = "string" } }; required = @("path") } } }
$toolAsk = @{ role = "user"; content = "Use the read_file tool to open README.md." }
Post "/v1/chat/completions" @{ model = $Model; messages = @($toolAsk); tools = @($tool); temperature = 0; max_tokens = 400 } "chat-tools.json"
Post "/v1/chat/completions" @{ model = $Model; messages = @($toolAsk); tools = @($tool); temperature = 0; max_tokens = 400; stream = $true } "chat-tools-stream.sse"

Save "v1-models-bad-route.json" (Get-Json "/v1/does-not-exist")

if (-not $wasLoaded) { & $lms unload $Model 2>&1 | Out-Null }
if (-not $wasRunning) { & $lms server stop 2>&1 | Out-Null }
$runtime = try { (Get-Content (Join-Path $out "api-v0-chat.json") -Raw | ConvertFrom-Json).runtime } catch { $null }
Save "captured.txt" ("captured {0} with {1}; engine {2} {3}" -f (Get-Date -Format s), $Model, $runtime.name, $runtime.version)
Get-ChildItem $out | Select-Object Name, Length
