# Records what llama.cpp's own server replies, before Squid's front door translates it.
# Tests replay it through a fake engine, so the translation is checked without a GPU.
#
#   pwsh -File golden\capture-llamacpp.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot
$out = Join-Path $PSScriptRoot "llamacpp"
New-Item -ItemType Directory -Force $out | Out-Null
$wasLoaded = (node "$repo\src\cli.ts" ps | Out-String) -match "qwen/qwen3.5-9b"
if (-not $wasLoaded) { node "$repo\src\cli.ts" load qwen/qwen3.5-9b --context 8192 | Out-Null }

$engine = Get-Content "$env:USERPROFILE\.squid\loaded\qwen%2Fqwen3.5-9b.json" | ConvertFrom-Json
$ask = @{ model = "qwen/qwen3.5-9b"; messages = @(@{ role = "user"; content = "Reply with exactly one word: pong" }); temperature = 0; max_tokens = 2000 }
# curl.exe writes each reply straight to its file, byte for byte.
function Post($body, $file) {
    $body | ConvertTo-Json -Depth 10 -Compress | curl.exe -s -N -X POST "http://127.0.0.1:$($engine.port)/v1/chat/completions" `
        -H "Content-Type: application/json" -H "Authorization: Bearer $($engine.apiKey)" --data-binary "@-" -o (Join-Path $out $file)
}
Post $ask "chat-nonstream.json"
Post ($ask + @{ stream = $true; stream_options = @{ include_usage = $true } }) "chat-stream-usage.sse"
# The same tool request LM Studio was recorded with (capture-lmstudio.ps1), streamed.
$tool = @{ type = "function"; function = @{
    name = "read_file"; description = "Read a text file from the current project"
    parameters = @{ type = "object"; properties = @{ path = @{ type = "string" } }; required = @("path") } } }
Post @{ model = "qwen/qwen3.5-9b"; messages = @(@{ role = "user"; content = "Use the read_file tool to open README.md." }); tools = @($tool); temperature = 0; max_tokens = 2000; stream = $true } "chat-tools-stream.sse"

if (-not $wasLoaded) { node "$repo\src\cli.ts" unload qwen/qwen3.5-9b | Out-Null }
Get-ChildItem $out | Select-Object Name, Length
