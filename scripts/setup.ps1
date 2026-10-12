param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (!$Python) { $Python = Join-Path $taskRoot '.tools\python311\python.exe' }
if (!(Test-Path -LiteralPath $Python)) {
    New-Item -ItemType Directory -Force '.tools' | Out-Null
    Invoke-WebRequest 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '.tools\python-installer.exe'
    $taskSignature = Get-AuthenticodeSignature '.tools\python-installer.exe'
    if ($taskSignature.Status -ne 'Valid' -or $taskSignature.SignerCertificate.Subject -notmatch 'Python Software Foundation') { throw 'Python installer signature verification failed' }
    $taskTarget = Join-Path $taskRoot '.tools\python311'
    $taskInstall = Start-Process '.tools\python-installer.exe' -ArgumentList @('/quiet','InstallAllUsers=0',('TargetDir="'+$taskTarget+'"'),'Include_launcher=0','InstallLauncherAllUsers=0','PrependPath=0','AssociateFiles=0','Shortcuts=0','Include_doc=0','Include_test=0') -WindowStyle Hidden -Wait -PassThru
    if ($taskInstall.ExitCode -notin 0,3010) { throw "Python installation failed: $($taskInstall.ExitCode)" }
}
& $Python -m venv .venv
if ($LASTEXITCODE) { throw 'venv failed' }
$taskVenvPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
& $taskVenvPython -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 required"'
if ($LASTEXITCODE) { throw 'Wrong Python version' }
& $taskVenvPython -m pip install setuptools==69.5.1 wheel==0.45.1 --index-url https://pypi.org/simple
if ($LASTEXITCODE) { throw 'Build helpers installation failed' }
& $taskVenvPython scripts/download-torch.py
if ($LASTEXITCODE) { throw 'PyTorch wheel download/checksum verification failed' }
& $taskVenvPython -m pip install --no-deps '.tools\wheels\torch-2.0.1+cu118-cp311-cp311-win_amd64.whl'
if ($LASTEXITCODE) { throw 'PyTorch installation failed' }
& $taskVenvPython -m pip install torchvision==0.15.2 --no-deps --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE) { throw 'PyTorch installation failed' }
$taskMmcv = Join-Path $taskRoot '.tools\wheels\mmcv-2.0.1-cp311-cp311-win_amd64.whl'
$taskMmcvExpected = 'ABD3B7EAD5774CA90E630603DE0E5DF961222CFE043F9E827177E614DB77C720'
if (!(Test-Path -LiteralPath $taskMmcv) -or (Get-FileHash -Algorithm SHA256 -LiteralPath $taskMmcv).Hash -ne $taskMmcvExpected) {
    & curl.exe --fail --location --retry 3 --connect-timeout 30 --output ($taskMmcv+'.download') 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv-2.0.1-cp311-cp311-win_amd64.whl'
    if ($LASTEXITCODE) { throw 'MMCV download failed' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath ($taskMmcv+'.download')).Hash -ne $taskMmcvExpected) { throw 'MMCV checksum mismatch' }
    Move-Item -LiteralPath ($taskMmcv+'.download') -Destination $taskMmcv -Force
}
& $taskVenvPython -m pip install -r requirements-lock.txt --find-links .tools/wheels --only-binary mmcv --no-build-isolation --index-url https://pypi.org/simple
if ($LASTEXITCODE) { throw 'Dependencies installation failed' }
& $taskVenvPython -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE) { throw 'Package installation failed' }
& $taskVenvPython scripts/setup-assets.py
if ($LASTEXITCODE) { throw 'Model assets setup failed' }
& $taskVenvPython scripts/gpu-smoke.py
if ($LASTEXITCODE) { throw 'GPU inference gate failed' }
& $taskVenvPython -m pip check
if ($LASTEXITCODE) { throw 'Dependency check failed' }
& $taskVenvPython -m pytest -q
if ($LASTEXITCODE) { throw 'Tests failed' }
