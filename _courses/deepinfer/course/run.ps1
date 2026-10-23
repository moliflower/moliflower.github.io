# Run the CPU-only course with a working Python 3.10+; installs nothing.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CourseArguments)

$ErrorActionPreference = 'Stop'
$candidates = @(
    @{ Command = 'python'; Prefix = @() },
    @{ Command = 'py'; Prefix = @('-3') },
    @{ Command = 'python3'; Prefix = @() },
    @{ Command = (Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'); Prefix = @() }
)
foreach ($candidate in $candidates) {
    $command = Get-Command $candidate.Command -ErrorAction SilentlyContinue
    if (-not $command) { continue }
    $prefix = $candidate.Prefix
    try {
        & $command.Source @prefix -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
    } catch { continue }
    $target = Join-Path $PSScriptRoot 'tools/grade.py'
    if ($CourseArguments.Count -eq 1 -and $CourseArguments[0] -eq '--demo') {
        $target = Join-Path $PSScriptRoot 'foundations/demo.py'
        $CourseArguments = @()
    }
    & $command.Source @prefix -B -X utf8 $target @CourseArguments
    exit $LASTEXITCODE
}
Write-Error 'No working Python 3.10+ was found. See course/foundations/README.md for setup.'
exit 1
