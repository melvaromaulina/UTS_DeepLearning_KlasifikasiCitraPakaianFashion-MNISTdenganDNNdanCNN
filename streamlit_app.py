"""Aplikasi demo klasifikasi pakaian Fashion-MNIST (CNN) dengan Streamlit.

Alur: foto -> praproses (identik dengan notebook) -> prediksi TFLite -> Grad-CAM (model Keras).
Model yang dipakai: CNN_C_BN_Dropout (436.650 parameter), akurasi uji 93,75%.
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Klasifikasi Pakaian Fashion-MNIST", page_icon="👕", layout="centered")


# ---------------------------------------------------------------- pemuatan model (sekali saja)
@st.cache_resource(show_spinner="Memuat model...")
def load_assets():
    info = json.loads((MODEL_DIR / "model_info.json").read_text())

    interp = tf.lite.Interpreter(model_path=str(MODEL_DIR / "model_final.tflite"))
    interp.allocate_tensors()

    keras_model = tf.keras.models.load_model(MODEL_DIR / "best_model.keras", compile=False)
    layer_name = info["gradcam_layer"]
    try:
        grad_model = tf.keras.Model(
            keras_model.inputs, [keras_model.get_layer(layer_name).output, keras_model.output]
        )
    except Exception:  # cadangan: rangkai ulang graf berantai seperti di notebook
        inp = tf.keras.Input(shape=tuple(info["input_shape"]))
        x, conv_out = inp, None
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
            if layer.name == layer_name:
                conv_out = x
        grad_model = tf.keras.Model(inp, [conv_out, x])
    return info, interp, grad_model


# ---------------------------------------------------------------- praproses (sama dengan notebook)
def preprocess_user_image(pil_img):
    """Foto bebas -> array (1, 28, 28, 1) float32 skala 0-255."""
    img = ImageOps.exif_transpose(pil_img)
    img = ImageOps.grayscale(img)
    if np.asarray(img).mean() > 127:          # latar terang -> inversi (Fashion-MNIST berlatar hitam)
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    side = max(img.size)
    img = ImageOps.pad(img, (side, side), color=0)   # jadikan persegi tanpa mengubah rasio
    img = img.resize((28, 28), Image.LANCZOS)
    return np.asarray(img, dtype="float32")[np.newaxis, ..., np.newaxis]


def predict_probs(interp, x):
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    return interp.get_tensor(out["index"])[0].copy()


def gradcam(grad_model, x, class_idx):
    xt = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(xt, training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights, axis=-1, keepdims=True))
    cam = tf.image.resize(cam, (28, 28), method="bilinear")[..., 0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()[0]


# ---------------------------------------------------------------- tampilan
info, interp, grad_model = load_assets()
class_names = info["class_names"]

st.title("👕 Klasifikasi Citra Pakaian")
st.caption(
    f"Model: CNN ({info['model_name']}) · dilatih pada Fashion-MNIST · "
    f"akurasi data uji {info['test_accuracy'] * 100:.2f}% · 10 kelas"
)

with st.expander("Tips foto agar hasil lebih baik"):
    st.markdown(
        "- Satu benda saja, **latar polos** (putih atau gelap), benda memenuhi sebagian besar foto.\n"
        "- Pakaian atasan: foto tampak depan, diratakan. Sepatu: **tampak samping**.\n"
        "- Model dilatih pada citra studio 28x28, jadi foto dengan latar ramai atau sudut tidak biasa "
        "bisa salah prediksi (pada uji foto sendiri kami, akurasinya 27 dari 50)."
    )

tab_up, tab_cam = st.tabs(["Unggah foto", "Kamera"])
with tab_up:
    uploaded = st.file_uploader("Pilih foto pakaian", type=["jpg", "jpeg", "png", "webp"])
with tab_cam:
    shot = st.camera_input("Ambil foto")

source = uploaded or shot
if source is None:
    st.info("Unggah atau ambil satu foto pakaian untuk memulai.")
    st.stop()

try:
    pil_img = Image.open(source)
    pil_img.load()
except Exception:
    st.error("File tidak bisa dibaca sebagai gambar. Coba file JPG atau PNG lain.")
    st.stop()

x = preprocess_user_image(pil_img)
probs = predict_probs(interp, x)
order = probs.argsort()[::-1]
top = int(order[0])

st.subheader(f"Prediksi: {class_names[top]} ({probs[top] * 100:.1f}%)")
if probs[top] < 0.6:
    st.warning("Keyakinan model rendah. Hasil ini sebaiknya tidak dianggap pasti.")

c1, c2, c3 = st.columns(3)
c1.image(pil_img, caption="Foto asli", use_container_width=True)
c2.image(x[0, :, :, 0].astype("uint8"), caption="Masukan 28x28 (yang dilihat model)",
         clamp=True, use_container_width=True)
cam = gradcam(grad_model, x, top)
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(x[0, :, :, 0], cmap="gray")
ax.imshow(cam, cmap="jet", alpha=0.45)
ax.axis("off")
c3.pyplot(fig, clear_figure=True)
c3.caption("Grad-CAM: area yang paling memengaruhi prediksi")

st.subheader("Tiga kemungkinan teratas")
st.bar_chart({class_names[i]: float(probs[i]) for i in order[:3]}, horizontal=True)

st.caption(
    "Prediksi memakai model TFLite (kuantisasi dynamic range, 442 KB). "
    "Grad-CAM dihitung dari model Keras yang bobotnya sama."
)
