<#
Race-weekend jobs that have to run from a home connection: F1's live-timing
server refuses GitHub's runners, so FastF1 can't load a session there.

  refresh_local.ps1 -Python <python.exe>            refresh the forecast (no-op outside a race weekend)
  refresh_local.ps1 -Python <python.exe> -Ingest    ingest finished races, pre-season testing pace and
                                                     the FIA's car-upgrade lists for Tuesday's retrain,
                                                     and score the forecasts published for them
  refresh_local.ps1 -Python <python.exe> -Register  install both as Windows scheduled tasks:
                                                     refresh every 30 min, ingest Mondays 15:00

Run it from a checkout of master (a separate git worktree keeps it away from
your own branches). It commits only what it generated, then pushes. Output
goes to %TEMP%\f1-refresh.log. The PC has to be on (or wake) for it to run;
a missed run starts as soon as the PC is back.

Alerts: when a job keeps failing (3 refreshes in a row, about 90 minutes,
since session data can legitimately take an hour to appear; or one Monday
ingest), it opens a GitHub issue with the end of the log, so GitHub emails
you. The issue is closed again by the next successful run. Needs `gh` logged
in; without it the job still runs, it just can't alert. A PC that is off
can't alert at all: .github/workflows/freshness.yml covers that from GitHub.
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
$job = if ($Ingest) { "ingest" } else { "refresh" }
$alertAfter = if ($Ingest) { 1 } else { 3 }
$title = if ($Ingest) { "Monday race ingest failed on the home PC" } else { "Race-weekend forecast refresh is failing on the home PC" }
$statePath = Join-Path $env:LOCALAPPDATA "f1-refresh-state.json"

function Read-State {
    $s = @{}
    try { (Get-Content $statePath -Raw | ConvertFrom-Json).psobject.Properties | ForEach-Object { $s[$_.Name] = $_.Value } } catch {}
    return $s
}
function Set-Failures([int]$n) {
    $s = Read-State
    $s[$job] = $n
    $s | ConvertTo-Json | Set-Content -Encoding utf8 $statePath
}
function Get-OpenAlert {
    try { return (gh issue list --state open --search "in:title `"$title`"" --json number --jq ".[0].number" 2>$null) } catch { return $null }
}
function Fail([int]$code) {
    $n = [int](Read-State)[$job] + 1
    Set-Failures $n
    "[$(Get-Date -Format s)] failed ($code), $n in a row" | Out-File -Append -Encoding utf8 $log
    if ($n -ge $alertAfter -and -not (Get-OpenAlert)) {
        $tail = (Get-Content $log -Tail 30 -Encoding utf8 | ForEach-Object { "    $_" }) -join "`n"
        $body = "The scheduled ``$job`` job on the home PC has failed $n time(s) in a row (exit code $code). " +
                "It keeps retrying; this issue closes itself on the next success.`n`nEnd of %TEMP%\f1-refresh.log:`n`n$tail"
        try { gh issue create --title $title --body $body 2>&1 | Out-File -Append -Encoding utf8 $log } catch {}
    }
    exit $code
}
function Step([string]$exe) {
    "[$(Get-Date -Format s)] $exe $args" | Out-File -Append -Encoding utf8 $log
    & $exe @args 2>&1 | Out-File -Append -Encoding utf8 $log
    if ($LASTEXITCODE) { Fail $LASTEXITCODE }
}

Step git pull --rebase --quiet
if ($Ingest) {
    $year = (Get-Date).ToUniversalTime().Year
    Step $Python -m src.data.ingest --seasons $year
    Step $Python -m src.data.fia --seasons $year
    Step $Python -m src.models.live_record
    $paths = @("data/raw/races", "data/raw/laps", "data/raw/testing", "data/raw/upgrades", "data/predictions/live_record.json")
    $message = "chore: ingest finished races"
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
if ([int](Read-State)[$job] -gt 0) {
    Set-Failures 0
    $open = Get-OpenAlert
    if ($open) { try { gh issue close $open --comment "Working again ($(Get-Date -Format s))." 2>&1 | Out-File -Append -Encoding utf8 $log } catch {} }
}
