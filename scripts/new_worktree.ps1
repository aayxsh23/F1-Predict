<#
A separate working folder for one agent session (or one piece of work), so two
sessions never edit the same files at once (it happened on 2026-10-19).

  scripts/new_worktree.ps1 -Name chat-fixes             # ../F1-chat-fixes on a new branch chat-fixes, from master
  scripts/new_worktree.ps1 -Name chat-fixes -Frontend   # also npm ci, for frontend work

Git gives the new folder only tracked files. The data git ignores (caches that
are slow or impossible to fetch again, the Wikipedia index, the chat's training
conversations, replay snapshots) is linked from this folder rather than copied,
so every worktree shares one copy. The feature table (data/processed) is built
per worktree, since a session may change features. Python is this folder's
.venv. When the work is merged: scripts/remove_worktree.ps1 -Name <Name>
(NOT `git worktree remove --force`, which follows the junctions and deletes
the shared data).
#>
param([Parameter(Mandatory = $true)][string]$Name, [switch]$Frontend)
# native tools write progress to stderr; judge them by exit code, not by stderr (Windows PowerShell)
$ErrorActionPreference = "Continue"
# the main folder (the one with .venv and the data), even when run from inside a worktree
$main = Split-Path (git -C $PSScriptRoot rev-parse --path-format=absolute --git-common-dir) -Parent
$dest = Join-Path (Split-Path $main -Parent) "F1-$Name"
git -C $main fetch --quiet origin
git -C $main worktree add $dest -b $Name --no-track origin/master
if ($LASTEXITCODE) { exit $LASTEXITCODE }

# The local-only docs (CLAUDE.md, AGENTS.md, PROGRESS.md, LEARNING.md, plans) are gitignored and live
# only in the main folder. Not hard-linked: editors replace files on save, which silently splits a hard
# link into two copies (tested 2026-10-21). The worktree gets pointers that load the real ones instead.
$pointer = @"
# Worktree $Name of the F1 project

This folder is a git worktree (branch ``$Name``). The project's instructions and logs live in the
MAIN folder only: read and edit them there, by these paths, never as copies here.

- $main\AGENTS.md (the rules), $main\CLAUDE.md, $main\GEMINI.md
- $main\PROGRESS.md and $main\LEARNING.md (update before ending a turn)
- $main\phase10-world-class-plan.md, $main\f1-race-predictor-explainer-project-plan.md

Remove this worktree only with ``scripts\remove_worktree.ps1 -Name $Name`` (never ``git worktree remove --force``:
it follows the data junctions and deletes the main folder's caches).

@$main\CLAUDE.md
@$main\AGENTS.md
"@
Set-Content -Encoding utf8 -Path (Join-Path $dest "CLAUDE.md") -Value $pointer
Set-Content -Encoding utf8 -Path (Join-Path $dest "GEMINI.md") -Value ($pointer -replace "(?m)^@.*\r?\n?", "")

# shared, git-ignored data: directories as junctions, single files as hard links (no admin rights needed)
$dirs = @("data/raw/fastf1_cache", "data/raw/fia_cache", "data/raw/jolpica_cache", "data/raw/wikipedia",
          "data/predictions/snapshots", "training/data")
foreach ($d in $dirs) {
    $src = Join-Path $main $d
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path (Split-Path (Join-Path $dest $d) -Parent) | Out-Null
        New-Item -ItemType Junction -Path (Join-Path $dest $d) -Target $src | Out-Null
    }
}
foreach ($f in @("data/corpus/wiki_index.json", ".env")) {
    $src = Join-Path $main $f
    if (Test-Path $src) { New-Item -ItemType HardLink -Path (Join-Path $dest $f) -Target $src | Out-Null }
}

$python = Join-Path $main ".venv\Scripts\python.exe"
Push-Location $dest
& $python -m src.features.build_dataset | Select-Object -Last 1
if ($Frontend) { Push-Location frontend; npm ci --silent; Pop-Location }
Pop-Location
Write-Output "ready: $dest (branch $Name). Python: $python"
