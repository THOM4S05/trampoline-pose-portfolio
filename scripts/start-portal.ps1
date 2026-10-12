param([string]$PublicUrl = '', [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$portalRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $portalRoot
$portalPython = Join-Path $portalRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $portalPython)) { throw 'Set up the tracker environment first; see README.md.' }
& $portalPython -c 'import flask, waitress'
if ($LASTEXITCODE) { throw 'Install the portal dependencies: .\.venv\Scripts\python.exe -m pip install -r requirements-web.txt' }
$portalArguments = @('-m', 'trampoline_pose.portal', '--port', $Port)
if ($PublicUrl) { $portalArguments += @('--public-url', $PublicUrl) }
& $portalPython @portalArguments
exit $LASTEXITCODE
