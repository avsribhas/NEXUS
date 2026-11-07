param(
    [Parameter(Mandatory)][ValidateSet('cpu','gpu')][string]$Kind,
    [Parameter(Mandatory)][string]$Session,
    [Parameter(Mandatory)][string]$CommandLine,
    [Parameter(Mandatory)][string]$Log
)
# Runs one command under C:\s9m\locks\<kind>.lock and always releases the lock. The rule is in CLAUDE.md, section 6.

$LockDir = 'C:\s9m\locks'
$Lock = Join-Path $LockDir "$Kind.lock"

Function Test-HolderAlive($HolderPid) {
    if (Get-Process -Id $HolderPid -ErrorAction SilentlyContinue) { return $true }
    wsl.exe -d Ubuntu -- kill -0 $HolderPid 2>$null
    return ($LASTEXITCODE -eq 0)
}

Function Enter-Lock {
    New-Item -ItemType Directory -Force -Path $LockDir | Out-Null
    while ($true) {
        try {
            $f = [IO.File]::Open($Lock, 'CreateNew', 'Write')
            $text = @{owner='dusk'; session=$Session; pid=$PID; kind=$Kind; start=(Get-Date).ToString('o')} | ConvertTo-Json -Compress
            $bytes = [Text.Encoding]::UTF8.GetBytes($text)
            $f.Write($bytes, 0, $bytes.Length)
            $f.Close()
            return
        } catch [System.IO.IOException] {
            $text = Get-Content $Lock -Raw -ErrorAction SilentlyContinue
            if ($text -match 'pid\D{0,5}(\d+)') {
                $holder = [int]$Matches[1]
                if (-not (Test-HolderAlive $holder)) {
                    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
                    continue
                }
            }
        }
        Start-Sleep -Seconds 30
    }
}

Enter-Lock
$code = 1
try {
    cmd.exe /c "$CommandLine > `"$Log`" 2>&1"
    $code = $LASTEXITCODE
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
    Set-Content -Path "$Log.done" -Value $code
}
exit $code
