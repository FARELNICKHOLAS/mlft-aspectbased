
# Aspect-Based Sentiment Analysis (ABSA) - TIX ID App Reviews

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

Proyek ini adalah sistem **Analisis Sentimen Berbasis Aspek (ABSA)** yang dirancang khusus untuk mengklasifikasikan ulasan pengguna aplikasi **TIX ID** di Google Play Store. Sistem ini menggunakan pendekatan *Deep Learning* dengan arsitektur **BiLSTM** dan representasi kata **FastText** untuk memberikan wawasan granular mengenai aspek layanan tertentu dan polaritas sentimennya.

## Fitur Utama
- **Single Prediction**: Input teks ulasan secara manual untuk mendapatkan hasil instan.
- **Batch Processing**: Unggah file CSV/Excel untuk menganalisis ribuan ulasan sekaligus.
- **Error Analysis**: Fitur evaluasi untuk membandingkan prediksi model dengan label asli (*ground truth*) lengkap dengan *Confusion Matrix*.
- **Indonesian NLP Pipeline**: Preprocessing lengkap (Cleaning, Normalisasi Slang, Stopwords, & Stemming Sastrawi).

---

## Arsitektur Model
Model dibangun menggunakan kombinasi:
1. **FastText Embedding**: Dimensi 300 untuk menangani kata-kata tidak baku (*slang*) dan *typo*.
2. **Bidirectional LSTM (BiLSTM)**: Menangkap konteks kalimat dari dua arah (maju dan mundur).
3. **SMOTE**: Penyeimbangan data untuk mengatasi *imbalance class* pada dataset asli.

Sistem mengklasifikasikan ulasan ke dalam **6 Aspek Utama**:
- Akses Akun
- Cakupan Layanan
- Layanan Tiket Bioskop
- Metode Pembayaran
- Pembaruan Aplikasi
- Promo dan Diskon

---

## Cara Menjalankan (Local Deployment)

### 1. Clone Repositori
```bash
git clone [https://github.com/FARELNICKHOLAS/mlft-aspectbased.git](https://github.com/FARELNICKHOLAS/mlft-aspectbased.git)
cd mlft-aspectbased

```

### 2. Buat dan Aktifkan Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Instal Dependensi

```bash
pip install -r requirements.txt

```

### 4. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py

```

---

## Hasil Evaluasi

Berdasarkan pengujian pada *test set*, model mencapai performa sebagai berikut:

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Sentiment** | 86% | 86% | 86% | 86% |
| **Aspect** | 88% | 81% | 82% | 82% |

---

## Struktur Direktori

```text
.
├── app.py                     # Script utama aplikasi Streamlit
├── bilstm_sentiment.h5        # Model Klasifikasi Sentimen
├── bilstm_aspect.h5           # Model Klasifikasi Aspek
├── tokenizer_sentiment.pkl    # Tokenizer untuk model sentimen
├── tokenizer_aspect.pkl       # Tokenizer untuk model aspek
├── label_encoder_aspect.pkl   # Label encoder untuk kategori aspek
├── merged_slang_dict.json     # Kamus normalisasi kata gaul
├── stopwords-id.txt           # Daftar kata berhenti (Indonesian)
├── requirements.txt           # Daftar library yang dibutuhkan
└── README.md

```

---

##  Anggota Kelompok (Kelompok 2)

1. **Wayan Farel Nickholas Sadewa** (2208561051)
2. **David Brave Moarota Zebua** (2208561063)
3. **I Made Treshnanda Mas** (2208561089)

**Dosen Pengampu:** Dr. Anak Agung Istri Ngurah Eka Karyawati, S.Si., M.Eng.
**Mata Kuliah:** Machine Learning For Text (MLFT) - Informatika Universitas Udayana.

```

---

### Tips Tambahan:
1. **File `requirements.txt`**: Pastikan Anda sudah membuat file ini di repositori Anda. Jika belum, jalankan `pip freeze > requirements.txt` saat virtual environment aktif.
2. **File Model**: Karena file `.h5` atau `.bin` (FastText) biasanya besar, pastikan Anda sudah mengunggahnya ke GitHub (jika di bawah 100MB) atau memberikan link eksternal jika menggunakan Git LFS.
3. **Screenshot**: Jika memungkinkan, tambahkan folder `assets/` dan masukkan gambar tampilan aplikasi Streamlit Anda, lalu panggil di README dengan `![alt text](assets/screenshot.png)` agar terlihat lebih menarik.



Apakah Anda ingin saya membantu membuatkan file **requirements.txt** atau bagian **Abstrak** versi bahasa Inggris untuk README ini?

```
