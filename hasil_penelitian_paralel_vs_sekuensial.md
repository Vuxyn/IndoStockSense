# Hasil Penelitian: Analisis Kinerja Pemrosesan Paralel vs Sekuensial

## 1. Pendahuluan
Bagian ini menyajikan hasil dari pengujian kinerja antara pemrosesan sekuensial dan paralel. Tujuannya adalah untuk mengukur efisiensi pemrosesan data dan waktu pencarian *hyperparameter* (Genetic Algorithm) dengan menggunakan pendekatan paralel pada PySpark.

## 2. Spesifikasi Lingkungan Pengujian
*   **Perangkat Keras (Hardware):**
    *   Device: CUDA (GPU Available)
*   **Perangkat Lunak (Software):**
    *   Framework/Library: PySpark (Versi 4.0.2)
*   **Dataset (IndoStockSense):**
    *   Total Cleaned Rows: 9.819 baris
    *   Train Size: 7.855
    *   Validation Size: 982
    *   Test Size: 982
    *   Distribusi Label: Netral (4.356), Positif (2.887), Negatif (2.576)

## 3. Hasil Kinerja Komputasi (Waktu Eksekusi)

Berikut adalah perbandingan waktu eksekusi yang tercatat selama proses:

| Tahapan Pemrosesan | Waktu Sekuensial (Detik) | Waktu Paralel (Detik) | Speedup (x) |
| :--- | :--- | :--- | :--- |
| **Data Processing** | *Data tidak tersedia (null)* | 20.65 | - |
| **Genetic Algorithm (GA)** | *Data tidak tersedia (null)* | 964.18 | - |

> **Catatan:** Waktu eksekusi sekuensial pada data *log* tercatat sebagai `null`. Jika pengujian sekuensial telah dilakukan secara terpisah, Anda dapat memperbarui angka di tabel ini untuk menghitung *speedup* secara pasti. Pengujian GA dilakukan dengan *Population Size* 20 dan 5 *Generations*.

## 4. Kinerja Model (IndoBERT-LoRA Sentiment Analysis)

Penggunaan Genetic Algorithm (GA) untuk optimasi *hyperparameter* menunjukkan peningkatan performa (F1-Score) dibandingkan dengan *baseline*:

| Skenario | Baseline F1-Score | Optimized F1-Score (GA) | Peningkatan (Improvement) |
| :--- | :--- | :--- | :--- |
| **Paralel (PySpark)** | 0.6831 | 0.8137 | + 0.1307 |
| **Sekuensial** | 0.6831 | 0.8149 | + 0.1318 |

**Hyperparameter Terbaik yang Ditemukan (Paralel):**
*   Learning Rate: `5e-05`
*   Batch Size: `8`
*   LoRA Rank: `16`
*   LoRA Alpha: `32`
*   LoRA Dropout: `0.1`
*   Weight Decay: `0.05`
*   Epochs: `3`

## 5. Kesimpulan
Berdasarkan hasil pengujian yang terekam:
1. Pemrosesan paralel dengan PySpark untuk pra-pemrosesan data berukuran ~10.000 baris memakan waktu sekitar **20.65 detik**.
2. Proses optimasi *hyperparameter* menggunakan Genetic Algorithm secara paralel berhasil diselesaikan dalam waktu **964.18 detik**.
3. Dari sisi kualitas model, optimasi dengan GA berhasil meningkatkan F1-Score secara signifikan (dari 0.68 menjadi 0.81). Baik eksekusi paralel maupun sekuensial menghasilkan kualitas model (*hyperparameter* optimal) yang hampir setara.
