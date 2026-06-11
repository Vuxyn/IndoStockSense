# ═══════════════════════════════════════════════════════════════
# START SPARK MASTER — Jalankan di PowerShell
# ═══════════════════════════════════════════════════════════════
# Cara pakai:
#   1. Buka PowerShell
#   2. cd ke folder proyek
#   3. conda activate spark312
#   4. .\start_master.ps1
# ═══════════════════════════════════════════════════════════════

$SPARK_HOME = "C:\Users\Mahesa\miniconda3\envs\spark312\Lib\site-packages\pyspark"
$MASTER_IP = "100.70.125.49"  # IP Tailscale laptop

$env:SPARK_HOME = $SPARK_HOME
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21"

Write-Host "==========================================="
Write-Host "  Starting Spark Master"
Write-Host "  SPARK_HOME : $SPARK_HOME"
Write-Host "  MASTER IP  : $MASTER_IP"
Write-Host "  MASTER URL : spark://${MASTER_IP}:7077"
Write-Host "  WEB UI     : http://${MASTER_IP}:8080"
Write-Host "==========================================="

& "$SPARK_HOME\bin\spark-class.cmd" org.apache.spark.deploy.master.Master --host $MASTER_IP --port 7077
