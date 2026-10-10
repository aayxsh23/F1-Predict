<#
Race-weekend jobs that have to run from a home connection: F1's live-timing
server refuses GitHub's runners, so FastF1 can't load a session there.

  refresh_local.ps1 -Python <python.exe>            refresh the forecast (no-op outside a race weekend)
  refresh_local.ps1 -Python <python.exe> -Ingest    ingest finished races, pre-season testing pace and
                                                     the FIA's car-upgrade lists for Tuesday's retrain
  refresh_local.ps1 -Python <python.exe> -Register  install both as Windows scheduled tasks:
                                                     refresh every 30 min, ingest Mondays 15:00

Run it from a checkout of master (a separate git worktree keeps it away from
your own branches). It commits only what it generated, then pushes. Output
goes to %TEMP%\f1-refresh.log. The PC has to be on (or wake) for it to run;
a missed run starts as soon as the PC is back.
#>
param([string]$Python = "python", [switch]$Ingest, [switch]$Register)

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$log = Join-Path $env:TEMP "f1-refresh.log"

if ($Register) {
    $ErrorActionPreference = "Stop"
    $run = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Python `"$Python`""
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 25)
    $every30 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 30)
    $monday = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "15:00"
    Register-ScheduledTask -TaskName "F1 forecast refresh" -Trigger $every30 -Settings $settings -Force `
        -Action (New-ScheduledTaskAction -Execute "powershell.exe" -Argument $run -WorkingDirectory $repo) | Out-Null
    Register-ScheduledTask -TaskName "F1 race ingest" -Trigger $monday -Settings $settings -Force `
        -Action (New-ScheduledTaskAction -Execute "powershell.exe" -Argument "$run -Ingest" -WorkingDirectory $repo) | Out-Null
    Write-Output "registered: F1 forecast refresh (every 30 min), F1 race ingest (Mondays 15:00)"
    return
}

# native tools write progress to stderr; judge them by exit code, not by stderr
$ErrorActionPreference = "Continue"
function Step([string]$exe) {
    "[$(Get-Date -Format s)] $exe $args" | Out-File -Append -Encoding utf8 $log
    & $exe @args 2>&1 | Out-File -Append -Encoding utf8 $log
    if ($LASTEXITCODE) { "[$(Get-Date -Format s)] failed ($LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log; exit $LASTEXITCODE }
}

Step git pull --rebase --quiet
if ($Ingest) {
    $year = (Get-Date).ToUniversalTime().Year
    Step $Python -m src.data.ingest --seasons $year
    Step $Python -m src.data.fia --seasons $year
    $paths, $message = @("data/raw/races", "data/raw/laps", "data/raw/testing", "data/raw/upgrades"), "chore: ingest finished races"
} else {
    Step $Python -m src.models.refresh_job --weekend-only
    $paths, $message = @("data/predictions"), "chore: refresh predictions"
}
Step git add @paths
git diff --cached --quiet
if ($LASTEXITCODE) {
    Step git commit --quiet -m $message
    Step git pull --rebase --quiet
    Step git push --quiet
}
