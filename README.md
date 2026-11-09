# Klasifikasi Citra Pakaian Fashion-MNIST dengan DNN dan CNN

Proyek Deep Learning kelompok: membandingkan **MLP (DNN) sebagai baseline**, **CNN buatan sendiri**, dan **transfer learning MobileNetV2** pada dataset Fashion-MNIST, lalu men-deploy model terbaik sebagai aplikasi web.

- **Aplikasi (demo):** [ISI LINK APLIKASI STREAMLIT]
- **Video demo + tahapan deployment:** [ISI LINK VIDEO]
- **Laporan teknis:** [ISI NAMA FILE / LINK LAPORAN]

**Kelompok:** [ISI NAMA KELOMPOK] · Anggota: [ISI NAMA DAN NIM ANGGOTA]

## Ringkasan hasil

| Item | Nilai |
|---|---|
| Dataset | Fashion-MNIST, 70.000 citra grayscale 28x28, 10 kelas |
| Pembagian data | latih 54.000 / validasi 6.000 / uji 10.000 (stratifikasi) |
| Model final | CNN `CNN_C_BN_Dropout` (BatchNorm + Dropout, tanpa augmentasi) |
| Parameter | 436.650 |
| Akurasi uji | **93,75%** (CI 95% bootstrap 93,25% - 94,20%), macro F1 0,9372 |
| Rata-rata 3 seed | 93,70% ± 0,24 |
| Baseline MLP | 89,30% ± 0,45 (McNemar vs CNN: p = 2,8e-67) |
| MobileNetV2 | 90,46% (1 seed, 2,27 juta parameter) |
| Model deployment | TFLite dynamic range, 442 KB (Keras 1.806 KB), akurasi uji 93,70% |
| Uji foto sendiri | 27 dari 50 benar (54%, CI 95% Wilson 40,4% - 67,0%) |

Angka di atas berasal dari satu kali run notebook (`ringkasan_angka_laporan.json`) dan sama dengan laporan dan video.

## Struktur repo

```
streamlit_app.py      # aplikasi web (Streamlit)
requirements.txt      # dependensi aplikasi
models/
  best_model.keras    # model Keras (dipakai untuk Grad-CAM)
  model_final.tflite  # model TFLite terkuantisasi (dipakai untuk prediksi)
  model_info.json     # nama kelas, layer Grad-CAM, metadata
Fashion_MNIST_DNN_CNN_tfdata.ipynb   # notebook pelatihan dan evaluasi (Google Colab)
figures/              # gambar untuk laporan
outputs/              # tabel hasil (csv/json), environment.json
```

## Cara menjalankan aplikasi secara lokal

Butuh Python 3.11 - 3.13.

```bash
git clone [ISI URL REPO]
cd [ISI NAMA REPO]
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Buka http://localhost:8501, unggah foto pakaian (latar polos, satu benda). Aplikasi menampilkan kelas prediksi, tiga kemungkinan teratas, masukan 28x28 yang dilihat model, dan peta panas Grad-CAM.

## Cara melatih ulang

1. Buka notebook di Google Colab, pilih Runtime > Change runtime type > GPU (T4).
2. Runtime > Run all. Eksperimen inti diulang pada 3 seed (`SEEDS`); ubah ke `[42]` bila waktu terbatas.
3. Hasil: `outputs/models/` (tiga file model), `figures/`, dan `outputs/*.csv|json`. Menjalankan ulang dapat mengubah angka sedikit karena GPU tidak sepenuhnya deterministik.
4. Uji foto sendiri membutuhkan folder `foto_uji/` (5 foto per kelas); sel tersebut bisa dilewati bila tidak ada.

## Deployment (Streamlit Community Cloud)

1. Repo ini Public dan berisi `streamlit_app.py` serta `requirements.txt` di root.
2. Di share.streamlit.io: Create app, pilih repo ini, branch `main`, main file `streamlit_app.py`.
3. Advanced settings: pilih Python 3.12 atau 3.13, lalu Deploy.

## Teknologi

Python, TensorFlow/Keras 2.21 (`tf.data`, TFLite), scikit-learn, NumPy, pandas, Matplotlib, Seaborn, Pillow, Streamlit. Pelatihan di Google Colab (GPU Tesla T4).

## Keterbatasan

- Model dilatih pada citra studio 28x28 berlatar hitam; pada foto sendiri akurasinya turun menjadi 54% (domain shift). Sneaker dan Ankle boot sering diprediksi sebagai Bag.
- Kelas Shirt paling sulit (recall 77,4%), sering tertukar dengan T-shirt/top, Coat, dan Pullover.
- Augmentasi tidak membantu pada eksperimen ini (varian D 92,10% vs varian C 93,70%).

## Dataset

Fashion-MNIST, Zalando Research (Xiao, Rasul, Vollgraf, 2017). Dimuat lewat `keras.datasets`.
