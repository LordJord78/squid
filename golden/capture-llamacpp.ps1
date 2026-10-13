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
$body = @{ model = "qwen/qwen3.5-9b"; messages = @(@{ role = "user"; content = "Reply with exactly one word: pong" }); temperature = 0; max_tokens = 2000 } | ConvertTo-Json -Depth 5 -Compress
$reply = $body | curl.exe -s -X POST "http://127.0.0.1:$($engine.port)/v1/chat/completions" -H "Content-Type: application/json" -H "Authorization: Bearer $($engine.apiKey)" --data-binary "@-"
Set-Content (Join-Path $out "chat-nonstream.json") $reply -Encoding utf8NoBOM

if (-not $wasLoaded) { node "$repo\src\cli.ts" unload qwen/qwen3.5-9b | Out-Null }
Get-ChildItem $out | Select-Object Name, Length
