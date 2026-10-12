$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskReport = [ordered]@{
    inspected_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    operating_system = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture
    cpu = Get-CimInstance Win32_Processor | Select-Object Name
    memory = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
    gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
    python_installations = (& py -0p 2>&1 | Out-String)
    ffmpeg = (& ffmpeg -version | Select-Object -First 1)
    ffprobe = (& ffprobe -version | Select-Object -First 1)
}
New-Item -ItemType Directory -Force (Join-Path $taskRoot 'reports') | Out-Null
$taskReport | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $taskRoot 'reports\hardware.json')
$taskReport | ConvertTo-Json -Depth 5
