# Registers a daily Windows Task Scheduler job that snapshots KSE-100 constituents after the PSX close.
# Run once from the repo root in PowerShell:  .\scripts\register_psx_snapshot_task.ps1
# Remove it later with:                       Unregister-ScheduledTask -TaskName "TRA PSX constituents"
param([string]$Time = "17:30")

$repo = (Resolve-Path "$PSScriptRoot\..").Path
$uv = (Get-Command uv).Source
$action = New-ScheduledTaskAction -Execute $uv `
    -Argument "run python -m tra.scrapers.psx.run --dataset constituents --index KSE100" `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "TRA PSX constituents" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Daily KSE-100 constituent snapshot (Multi-Agent Trading Research System)" -Force
