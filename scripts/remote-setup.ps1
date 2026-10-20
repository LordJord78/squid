# Remote mode (step 22), run once on the PC: a scheduled task, "Squid server", that starts Squid's
# server as you, without admin rights and without a window. The Mac runs it over SSH with
#   schtasks /run /tn "Squid server"
# Why not `squid server start` over SSH: on Windows a program started from an SSH session keeps that
# session open until it ends, and runs elevated, where your own terminal can't stop it.
#
#   pwsh -File scripts\remote-setup.ps1            (from an admin terminal, or over SSH)
#   Unregister-ScheduledTask "Squid server"         undoes it
$ErrorActionPreference = "Stop"
$cli = (Resolve-Path (Join-Path $PSScriptRoot "..\src\cli.ts")).Path
$node = (Get-Command node).Source
$action = New-ScheduledTaskAction -Execute $node -Argument "`"$cli`" server start" -WorkingDirectory $HOME
# S4U: runs as you whether or not you're logged on, with no window and no stored password. Limited:
# not elevated, the same rights as your own terminal.
$me = [Security.Principal.WindowsIdentity]::GetCurrent().Name # BLD\jorda; $env:USERDOMAIN is unreliable over SSH
$principal = New-ScheduledTaskPrincipal -UserId $me -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0 -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "Squid server" -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Task 'Squid server' runs: $node `"$cli`" server start"
