# ============================================================
# spark-master-firewall.ps1
# Setup Windows Firewall untuk Spark Master (Tailscale cluster)
# Jalankan sebagai Administrator!
# ============================================================

#Requires -RunAsAdministrator

$RuleName_Prefix = "SparkMaster"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Spark Master Firewall Setup (Windows)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Daftar port yang perlu dibuka
$ports = @(
    @{ Port = 7077;        Desc = "Spark Master (Worker registration)" },
    @{ Port = 8080;        Desc = "Spark Master Web UI" },
    @{ Port = 4040;        Desc = "Spark App Web UI" },
    @{ Port = 7078;        Desc = "Spark Master REST port" },
    @{ Port = 6066;        Desc = "Spark Master REST submission" }
)

# Range ephemeral port untuk executor/driver callback
$ephemeralStart = 49152
$ephemeralEnd   = 65535

# ---- Buka port satuan ----
foreach ($p in $ports) {
    $name = "$RuleName_Prefix-$($p.Port)"

    # Hapus rule lama kalau ada
    if (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue) {
        Remove-NetFirewallRule -DisplayName $name
    }

    New-NetFirewallRule `
        -DisplayName  $name `
        -Direction    Inbound `
        -Protocol     TCP `
        -LocalPort    $p.Port `
        -Action       Allow `
        -Profile      Any `
        -Description  $p.Desc | Out-Null

    Write-Host "  [OK] Port $($p.Port) dibuka  — $($p.Desc)" -ForegroundColor Green
}

# ---- Buka range ephemeral ----
$epName = "$RuleName_Prefix-Ephemeral"
if (Get-NetFirewallRule -DisplayName $epName -ErrorAction SilentlyContinue) {
    Remove-NetFirewallRule -DisplayName $epName
}
New-NetFirewallRule `
    -DisplayName  $epName `
    -Direction    Inbound `
    -Protocol     TCP `
    -LocalPort    "$ephemeralStart-$ephemeralEnd" `
    -Action       Allow `
    -Profile      Any `
    -Description  "Spark executor/driver callback ports" | Out-Null

Write-Host "  [OK] Ephemeral ports $ephemeralStart-$ephemeralEnd dibuka" -ForegroundColor Green

# ---- Set env SPARK_LOCAL_IP ke IP Tailscale ----
Write-Host ""
Write-Host "Mendeteksi IP Tailscale..." -ForegroundColor Cyan

$tailscaleIP = $null
try {
    $tsOutput = & tailscale ip -4 2>&1
    if ($LASTEXITCODE -eq 0) {
        $tailscaleIP = $tsOutput.Trim()
    }
} catch {
    Write-Host "  [WARN] tailscale.exe tidak ditemukan di PATH" -ForegroundColor Yellow
}

if ($tailscaleIP) {
    Write-Host "  [OK] IP Tailscale: $tailscaleIP" -ForegroundColor Green

    # Set environment variable untuk sesi ini
    $env:SPARK_LOCAL_IP = $tailscaleIP
    Write-Host "  [OK] SPARK_LOCAL_IP=$tailscaleIP (sesi ini)" -ForegroundColor Green

    # Tanya user mau simpan permanent gak
    $save = Read-Host "`nSimpan SPARK_LOCAL_IP ke environment variable permanen? (y/n)"
    if ($save -eq "y") {
        [System.Environment]::SetEnvironmentVariable("SPARK_LOCAL_IP", $tailscaleIP, "User")
        Write-Host "  [OK] Disimpan ke User environment variables" -ForegroundColor Green
    }
} else {
    Write-Host "  [WARN] Tidak bisa auto-detect IP Tailscale." -ForegroundColor Yellow
    Write-Host "         Set manual: `$env:SPARK_LOCAL_IP = '100.x.x.x'" -ForegroundColor Yellow
}

# ---- Test koneksi Tailscale ke worker ----
Write-Host ""
$testPing = Read-Host "Test ping ke Colab worker? Masukkan IP Tailscale worker (kosongkan untuk skip)"
if ($testPing -ne "") {
    Write-Host "Pinging $testPing..." -ForegroundColor Cyan
    $pingResult = Test-Connection -ComputerName $testPing -Count 3 -ErrorAction SilentlyContinue
    if ($pingResult) {
        $avg = ($pingResult | Measure-Object -Property ResponseTime -Average).Average
        Write-Host "  [OK] Reachable! Avg latency: $([math]::Round($avg, 1)) ms" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Tidak bisa ping $testPing" -ForegroundColor Red
        Write-Host "         Pastikan Tailscale aktif di kedua sisi" -ForegroundColor Yellow
    }
}

# ---- Summary ----
Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Setup selesai. Ringkasan rule yang dibuat:" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Get-NetFirewallRule -DisplayName "$RuleName_Prefix*" | 
    Select-Object DisplayName, Enabled, Direction |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Selanjutnya:" -ForegroundColor Yellow
Write-Host "  1. Start Spark Master:" -ForegroundColor White
Write-Host "     `$env:SPARK_LOCAL_IP='$tailscaleIP'" -ForegroundColor Gray
Write-Host "     spark-class org.apache.spark.deploy.master.Master --host $tailscaleIP" -ForegroundColor Gray
Write-Host "  2. Jalankan sel di Colab untuk connect worker" -ForegroundColor White
Write-Host "  3. Cek di http://${tailscaleIP}:8080 apakah worker muncul" -ForegroundColor White
Write-Host ""
