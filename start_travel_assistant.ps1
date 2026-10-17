#requires -Version 5.1

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RagDir = Join-Path $ProjectRoot "rag-system"
$ModelPath = Join-Path $ProjectRoot "models\qwen3-0.6b-existing-data-merged"
$HealthUrl = "http://127.0.0.1:8010/health"
$PowerShellExe = (Get-Command powershell.exe).Source

function Resolve-QwenPython {
    if ($env:CONDA_DEFAULT_ENV -eq "qwen3-sft" -and $env:CONDA_PREFIX) {
        $currentEnvPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $currentEnvPython) {
            return $currentEnvPython
        }
    }

    $condaCandidates = @()
    foreach ($name in @("conda.exe", "conda.bat")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source) {
            $condaCandidates += $cmd.Source
        }
    }
    if ($env:CONDA_EXE) {
        $condaCandidates += $env:CONDA_EXE
    }
    # 双击 bat 时使用 -NoProfile，conda init 写在用户配置里，不会进入 PATH。
    $profilePaths = @(
        (Join-Path $env:USERPROFILE "Documents\WindowsPowerShell\profile.ps1"),
        (Join-Path $env:USERPROFILE "Documents\PowerShell\profile.ps1")
    )
    foreach ($profilePath in $profilePaths) {
        if (-not (Test-Path -LiteralPath $profilePath)) {
            continue
        }
        $profileText = Get-Content -LiteralPath $profilePath -Raw -ErrorAction SilentlyContinue
        foreach ($match in [regex]::Matches($profileText, '"([^"]*conda\.exe)"')) {
            $condaCandidates += $match.Groups[1].Value
        }
    }

    foreach ($condaExe in ($condaCandidates | Select-Object -Unique)) {
        if (-not $condaExe -or -not (Test-Path -LiteralPath $condaExe)) {
            continue
        }
        $condaRoot = Split-Path -Parent (Split-Path -Parent $condaExe)
        $envPython = Join-Path $condaRoot "envs\qwen3-sft\python.exe"
        if (Test-Path -LiteralPath $envPython) {
            return $envPython
        }
        $condaBase = (& $condaExe info --base 2>$null | Select-Object -Last 1)
        if ($condaBase) {
            $condaBase = $condaBase.Trim()
            $envPython = Join-Path $condaBase "envs\qwen3-sft\python.exe"
            if (Test-Path -LiteralPath $envPython) {
                return $envPython
            }
        }
    }

    throw "找不到 qwen3-sft 环境中的 python.exe。请先创建 qwen3-sft 环境，或检查 Conda 安装路径。"
}

function Test-RagHealthy {
    try {
        $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3
        return ($health.status -eq "healthy")
    }
    catch {
        return $false
    }
}

if (-not (Test-Path $RagDir)) {
    throw "RAG 目录不存在: $RagDir"
}
if (-not (Test-Path (Join-Path $RagDir "rag_api.py"))) {
    throw "找不到 RAG 启动文件: $RagDir\rag_api.py"
}
if (-not (Test-Path (Join-Path $ProjectRoot "travel_assistant_funcall_fixed.py"))) {
    throw "找不到旅行助手入口: $ProjectRoot\travel_assistant_funcall_fixed.py"
}
if (-not (Test-Path $ModelPath)) {
    throw "找不到微调后的合并模型: $ModelPath"
}

$PythonExe = Resolve-QwenPython
Write-Host "Python: $PythonExe" -ForegroundColor DarkGray
Write-Host "Model : $ModelPath" -ForegroundColor DarkGray

if (Test-RagHealthy) {
    Write-Host "RAG 已经运行，复用 http://127.0.0.1:8010" -ForegroundColor Green
}
else {
    Write-Host "正在启动 RAG 服务..." -ForegroundColor Cyan
    $ragCommand = @"
`$Host.UI.RawUI.WindowTitle = '旅行助手 - RAG 服务'
Set-Location -LiteralPath '$RagDir'
& '$PythonExe' '.\rag_api.py'
"@

    Start-Process `
        -FilePath $PowerShellExe `
        -WorkingDirectory $RagDir `
        -ArgumentList @("-NoLogo", "-NoExit", "-Command", $ragCommand) `
        -WindowStyle Normal | Out-Null

    $ragReady = $false
    for ($i = 1; $i -le 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-RagHealthy) {
            $ragReady = $true
            break
        }
        Write-Host "." -NoNewline
    }
    Write-Host ""
    if (-not $ragReady) {
        throw "RAG 服务在 30 秒内没有通过健康检查，请查看 RAG PowerShell 窗口。"
    }
    Write-Host "RAG 已就绪: $HealthUrl" -ForegroundColor Green
}

$assistantCommand = @"
`$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(`$false)
`$env:TRAVEL_ASSISTANT_BACKEND = 'local_qwen'
`$env:LOCAL_QWEN_MODEL_PATH = '$ModelPath'
Set-Location -LiteralPath '$ProjectRoot'
& '$PythonExe' '.\travel_assistant_funcall_fixed.py'
"@

Write-Host "正在打开微调后的旅行助手 PowerShell 窗口..." -ForegroundColor Cyan
Start-Process `
    -FilePath $PowerShellExe `
    -WorkingDirectory $ProjectRoot `
    -ArgumentList @("-NoLogo", "-NoExit", "-Command", $assistantCommand) `
    -WindowStyle Normal | Out-Null

Write-Host "启动完成。" -ForegroundColor Green
Write-Host "模型: $ModelPath"
Write-Host "在新窗口出现“您:”后输入旅行问题即可。"
