# ═══════════════════════════════════════════════════════════════════════════════
# COLAB WORKER SETUP — Copy-paste semua sel ini ke Google Colab
# ═══════════════════════════════════════════════════════════════════════════════
# Jalankan di 2 sesi Colab (1 biasa + 1 incognito dengan akun berbeda)
# Pastikan runtime = GPU (T4)
# ═══════════════════════════════════════════════════════════════════════════════

# %% [markdown]
# # 🔧 Spark Worker Setup — Colab
# Jalankan sel-sel berikut secara berurutan.
# **JANGAN** menutup atau menghentikan Sel terakhir (Spark Worker) selama eksperimen berjalan.

# %%
# ═══════════════════════════════════════════════════
# SEL 1: CEK ENVIRONMENT
# ═══════════════════════════════════════════════════
import sys
print(f"Python version: {sys.version}")

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
# ═══════════════════════════════════════════════════
# SEL 2: INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════
# PENTING: Versi PySpark HARUS 3.5.6 (sama dengan Master/Laptop)
!pip install -q pyspark==3.5.6
!pip install -q transformers datasets accelerate evaluate peft scikit-learn tqdm

# %%
# ═══════════════════════════════════════════════════
# SEL 3: INSTALL & CONNECT TAILSCALE
# ═══════════════════════════════════════════════════
!curl -fsSL https://tailscale.com/install.sh | sh

import subprocess, time

# Jalankan Tailscale daemon di background
subprocess.Popen(
    ["tailscaled", "--tun=userspace-networking", "--socks5-server=localhost:1055"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(5)  # Tunggu daemon siap

# ╔═══════════════════════════════════════════════════════════════╗
# ║  GANTI AUTH_KEY dan WORKER_NAME di bawah ini!                ║
# ║                                                              ║
# ║  Worker 1 (Colab biasa):    hostname = "colab-worker-1"      ║
# ║  Worker 2 (Colab incognito): hostname = "colab-worker-2"     ║
# ╚═══════════════════════════════════════════════════════════════╝

AUTH_KEY     = "tskey-auth-GANTI_DENGAN_AUTH_KEY_MU"
WORKER_NAME  = "colab-worker-1"    # Ganti jadi "colab-worker-2" untuk Colab kedua

!tailscale up --authkey="{AUTH_KEY}" --hostname="{WORKER_NAME}"

print("\n=== Tailscale Status ===")
!tailscale ip -4

# Ping Master untuk verifikasi koneksi
MASTER_IP = "100.68.30.53"
print(f"\n=== Ping Master ({MASTER_IP}) ===")
!tailscale ping {MASTER_IP} --c 3

# %%
# ═══════════════════════════════════════════════════
# SEL 4: PRE-DOWNLOAD MODEL & DATASET
# ═══════════════════════════════════════════════════
# Penting: cache model dan dataset SEBELUM Worker berjalan
# agar evaluasi fitness tidak perlu download ulang setiap kali

import os, urllib.request

# Download dataset
os.makedirs("data/raw", exist_ok=True)
DATA_URL  = "https://raw.githubusercontent.com/Vuxyn/IndoStockSense/main/data/raw/Dataset-CNBCI-Sentimented.csv"
DATA_PATH = "data/raw/Dataset-CNBCI-Sentimented.csv"
if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"✅ Dataset saved to {DATA_PATH}")
else:
    print("✅ Dataset already cached")

# Pre-cache IndoBERT model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "indobenchmark/indobert-base-p1"
print("\nDownloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Downloading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
)
del model  # Hapus dari memori GPU, sudah ter-cache di disk
print("✅ Model cached!")

# %%
# ═══════════════════════════════════════════════════
# SEL 5: JALANKAN SPARK WORKER
# ═══════════════════════════════════════════════════
# ⚠️  SEL INI AKAN BLOCKING (terus berjalan)
# ⚠️  JANGAN hentikan sel ini selama eksperimen berlangsung!

import subprocess

# Cari path spark-class
result = subprocess.run(
    ["find", "/", "-name", "spark-class", "-path", "*/pyspark/*", "-not", "-name", "*.cmd"],
    capture_output=True, text=True, timeout=15
)
SPARK_CLASS = result.stdout.strip().split("\n")[0]
print(f"spark-class: {SPARK_CLASS}")

# Dapatkan IP Tailscale Colab ini
result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True)
WORKER_IP = result.stdout.strip()
print(f"Worker IP  : {WORKER_IP}")

MASTER_IP = "100.68.30.53"  # IP Tailscale laptop
print(f"Master URL : spark://{MASTER_IP}:7077")
print()
print("=" * 50)
print("  🚀 Starting Spark Worker...")
print("  Sel ini akan terus berjalan. Jangan dihentikan!")
print("=" * 50)

# Jalankan Worker — BLOCKING
!{SPARK_CLASS} org.apache.spark.deploy.worker.Worker \
    spark://{MASTER_IP}:7077 \
    --host {WORKER_IP} \
    --cores 2 \
    --memory 10g
