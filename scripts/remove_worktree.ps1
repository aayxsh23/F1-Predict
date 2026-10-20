<#
Remove a worktree made by new_worktree.ps1, safely:

  scripts/remove_worktree.ps1 -Name chat-fixes

Its shared data folders are junctions into the main folder, and
`git worktree remove --force` FOLLOWS them and deletes the real data (tested
2026-10-21: FastF1's cache would have gone). So the junctions are unlinked
first (rmdir without /s removes only the link), then git removes the folder.
The branch is deleted only if it is merged into master.
#>
param([Parameter(Mandatory = $true)][string]$Name)
$ErrorActionPreference = "Continue"
# the main folder (the one with .venv and the data), even when run from inside a worktree
$main = Split-Path (git -C $PSScriptRoot rev-parse --path-format=absolute --git-common-dir) -Parent
$dest = Join-Path (Split-Path $main -Parent) "F1-$Name"
if (-not (Test-Path $dest)) { Write-Output "no worktree at $dest"; exit 1 }
Get-ChildItem $dest -Recurse -Force -Attributes ReparsePoint -ErrorAction SilentlyContinue | ForEach-Object {
    cmd /c rmdir "$($_.FullName)"   # the link only, never its contents
}
git -C $main worktree remove $dest   # no --force: refuses if anything uncommitted is left
if ($LASTEXITCODE) { Write-Output "not removed: commit or discard the changes in $dest first"; exit 1 }
git -C $main branch -d $Name 2>&1 | Out-Null
if ($LASTEXITCODE) { Write-Output "worktree removed; branch $Name kept (not merged into master)" } else { Write-Output "removed $dest and branch $Name" }
