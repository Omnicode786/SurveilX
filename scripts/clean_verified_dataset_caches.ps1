$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$journalPath = Join-Path $workspace 'reports/g4-cleanup-journal.json'
$journal = Get-Content -LiteralPath $journalPath -Raw | ConvertFrom-Json
if ($journal.deleted) { throw 'Journal already records deletion; do not repeat.' }
if (-not $journal.verification_complete) { throw 'Verification did not complete; retain pending directories.' }
if (-not $journal.yaml_configuration_verified) { throw 'Dataset configuration is not verified.' }
$trainingProcesses = & (Join-Path $workspace '.venv/Scripts/python.exe') -c 'import psutil,json; print(json.dumps([p.pid for p in psutil.process_iter(["name","cmdline"]) if "python" in (p.info["name"] or "").lower() and any(x.startswith(("scripts.train_","scripts.queue_","training.")) for x in (p.info["cmdline"] or []))]))'
if ($LASTEXITCODE -ne 0 -or $trainingProcesses.Trim() -ne '[]') { throw 'Cannot clean while a training/queue process may be running.' }
$runsRoot = (Resolve-Path -LiteralPath (Join-Path $workspace 'data/runs')).Path
$pendingEntries = @($journal.directories | Where-Object { -not $_.deleted })
# Verify every target and inventory before deleting any directory. Use native
# PowerShell path checks and deletion in this same shell, never cmd or a second shell.
foreach ($entry in $pendingEntries) {
    $target = (Resolve-Path -LiteralPath $entry.directory).Path
    if (-not $entry.verified_reproducible -or -not $target.StartsWith($runsRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $target) -ne 'dataset' -or (Split-Path -Parent (Split-Path -Parent $target)) -ne $runsRoot) { throw "Unsafe target: $target" }
    foreach ($parent in @($runsRoot, (Split-Path -Parent $target), $target)) {
        if ((Get-Item -LiteralPath $parent).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked directory: $parent" }
    }
    $contents = @(Get-ChildItem -LiteralPath $target -Recurse -Force)
    if (@($contents | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw "Linked content: $target" }
    if (@($contents | Where-Object { -not $_.PSIsContainer }).Count -ne $entry.inventory.Count) { throw "Inventory changed: $target" }
    foreach ($file in $entry.inventory) {
        $path = [IO.Path]::GetFullPath((Join-Path $target $file.path))
        if (-not $path.StartsWith($target + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Inventory path escaped target.' }
        if ((Get-Item -LiteralPath $path).Length -ne $file.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $file.sha256) { throw "Cache changed: $path" }
    }
    if ((Get-FileHash -LiteralPath $entry.source_manifest -Algorithm SHA256).Hash -ne $entry.source_manifest_sha256) { throw 'Source manifest changed.' }
}
foreach ($entry in $pendingEntries) {
    Remove-Item -LiteralPath $entry.directory -Recurse -Force
    if (Test-Path -LiteralPath $entry.directory) { throw 'Deletion incomplete.' }
    $entry | Add-Member -NotePropertyName deleted -NotePropertyValue $true -Force
    $journal | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $journalPath -Encoding utf8
}
$journal.deleted = $true
$journal | Add-Member -NotePropertyName completed_at -NotePropertyValue (Get-Date -Format o) -Force
$journal | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $journalPath -Encoding utf8
Write-Output "Removed $($journal.directories.Count) verified derived caches; $($journal.bytes_verified) bytes. Sources and model artifacts retained."
