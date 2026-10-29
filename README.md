# 🎙️ Hening - Real-Time Audio Noise Classifier (Docker & Flask Microservice)

Sistem pemantauan dan klasifikasi kebisingan ruangan laboratorium secara *real-time / live listening* menggunakan model **Support Vector Machine (SVM RBF)**, backend **Flask (Single Port 5000)**, dan kontainerisasi **Docker & Docker Compose**.

Sistem ini didesain khusus untuk mendengarkan audio secara langsung dari mikrofon/audio stream tanpa perlu upload file, serta mengevaluasi ambang batas **konsistensi ramai 10–15 detik** sebelum memicu alarm peringatan.

---

## 📁 Struktur Direktori Project `sistem_klasifikasi_ensamble/`

```text
sistem_klasifikasi_ensamble/
├── Dockerfile                      # Spesifikasi build container (Python 3.12-slim, PortAudio, ALSA)
├── docker-compose.yml              # Konfigurasi orkestrasi container (Single port 5000)
├── requirements.txt                # Dependensi minimal: Flask, Librosa, Scikit-Learn, SoundDevice
├── app.py                          # Backend Flask: Background audio capture, SVM inference, SSE streaming
├── templates/
│   └── index.html                  # Dashboard web modern: Live status, VU meter, 10-15s tracker, Web Audio API
├── svm_audio_model.pkl             # Model produksi akhir Support Vector Machine (RBF Kernel)
├── ensemble_audio_model.pkl        # Model pembanding Ensemble Learning (Soft Voting)
├── audio_scaler.pkl                # StandardScaler terlatih untuk 16 fitur akustik
├── feature_names.pkl               # Daftar 16 fitur akustik (RMS, Centroid, ZCR, MFCC 1-13)
├── model_metrics.json              # Rekapitulasi metrik evaluasi model (Akurasi, Presisi, Overfit Gap)
├── peringatan.wav                  # File audio nada sirine alarm
├── ekstraksi_audio_ensemble.ipynb  # Notebook riset & pemodelan lengkap
├── dataset_audio_final.csv         # Dataset 200 sampel audio UTM
├── fitur_mp3_berisik.csv           # Ekstraksi fitur 100 sampel berisik
├── fitur_mp3_non_berisik.csv       # Ekstraksi fitur 100 sampel non-berisik
└── README.md                       # Dokumentasi arsitektur dan panduan deployment
```

---

## 🐳 Persyaratan & Arsitektur Deployment Docker

1. **Hanya Membuka 1 Port (Single Port 5000):**
   - Seluruh layanan (halaman web HTML/CSS/JS, endpoint Server-Sent Events `/stream`, REST API kontrol `/api/control`, dan stream audio dari browser `/api/push_audio`) berjalan melalui satu port HTTP terpadu: `5000:5000`.
2. **Base Image Optimal (`python:3.12-slim`):**
   - Menggunakan Debian Bookworm minimal dengan dependensi sistem audio esensial: `portaudio19-dev`, `libasound2-dev`, `libsndfile1`, `alsa-utils`, dan `ffmpeg`.
3. **Dual-Engine Audio Listening (Resilient Architecture):**
   - **Mode 1 - Host SoundDevice:** Backend mencoba membuka `sounddevice.InputStream` secara otomatis jika container mendeteksi perangkat audio fisik (misal via device pass-through `/dev/snd` pada host Linux).
   - **Mode 2 - Web Audio API (Browser Mic Bridge):** Disediakan tombol *"Aktifkan Mic Browser"* pada antarmuka web yang memanfaatkan Web Audio API client (`navigator.mediaDevices.getUserMedia`) untuk mengalirkan audio mikrofon pengguna langsung ke server. Ini mengatasi kendala keterbatasan akses mikrofon host pada Docker Desktop di Windows/macOS.
   - **Mode 3 - Safe Stream Simulation:** Jika dijalankan di container tanpa input audio sama sekali, sistem tetap aktif dan tidak crash (*zero-downtime*).

---

## 🚀 Panduan Menjalankan dengan Docker

### 1. Prasyarat
Pastikan Docker dan Docker Compose telah terpasang di komputer Anda:
```bash
docker --version
docker compose version
```

### 2. Build dan Jalankan Container
Buka terminal, arahkan ke direktori `sistem_klasifikasi_ensamble`:
```bash
cd sistem_klasifikasi_ensamble
docker compose up --build
```

Opsi latar belakang (detached mode):
```bash
docker compose up --build -d
```

### 3. Akses Dashboard Web
Buka peramban (browser) dan akses:
```text
http://localhost:5000
```

### 4. Menghentikan Container
Untuk mematikan container:
```bash
docker compose down
```

---

## 💻 Menjalankan Secara Lokal (Tanpa Docker)

Jika ingin menjalankan aplikasi secara langsung di Python host lokal:

```bash
# 1. Pindah ke direktori project
cd sistem_klasifikasi_ensamble

# 2. Pasang dependensi pustaka
pip install -r requirements.txt

# 3. Jalankan server Flask
python app.py
```
Akses di browser melalui: `http://localhost:5000`.

---

## ⏱️ Mekanisme Deteksi Konsistensi Ramai 10–15 Detik

- **Masalah:** Suara keras sesaat (seperti batuk, benda terjatuh, pintu tertutup selama 1–2 detik) menghasilkan RMS tinggi namun **bukan** suasana ramai yang sebenarnya.
- **Solusi:** Sistem menerapkan pelacak durasi kebisingan kontinu:
  - Setiap detik berurutan dengan klasifikasi `BERISIK` menambah counter durasi ($+1\text{ detik}$).
  - Jika kebisingan mereda, counter berkurang secara adaptif.
  - **Ambang Batas Alarm:** Jika kebisingan bertahan $\ge 10\text{ detik}$ (hingga 15 detik), status darurat aktif, banner merah menyala, dan file audio `peringatan.wav` dibunyikan secara otomatis.

---

## 🔬 Ekstraksi 16 Fitur & Alasan Tanpa Seleksi Fitur

Sistem mengekstrak **16 fitur akustik** secara konsisten ($f_s = 22050\text{ Hz}$):
1. `rms`: Root Mean Square (Daya sinyal / intensitas).
2. `centroid`: Spectral Centroid (Titik berat spektrum frekuensi).
3. `zcr`: Zero Crossing Rate (Laju pergantian tanda gelombang).
4. `mfcc_1` s.d. `mfcc_13`: Koefisien Mel-Frekuensi untuk mengenali timbre vokal manusia.

### Mengapa Tidak Dilakukan Seleksi Fitur?
1. **Penurunan Akurasi Uji:** Pemangkasan fitur terbukti menurunkan akurasi generalisasi data uji dari **80.00% (16 fitur)** menjadi **72.50% - 75.00% (< 12 fitur)**.
2. **Integritas Karakteristik Akustik:** Seluruh 16 fitur membawa informasi penting yang saling melengkapi antara daya, kecerahan spektral, keacakan derau, dan karakteristik vokal.
3. **Dimensi Ringan:** 16 fitur numerik sangat ringan untuk komputasi real-time (< 20 ms) dan bebas risiko *curse of dimensionality*.
4. **Model Akhir:** **SVM (RBF Kernel)** terpilih sebagai model akhir karena akurasi uji terbaik (**80.00%**), presisi tertinggi (**87.50%**), dan *overfitting gap* terendah (**14.38%**).
