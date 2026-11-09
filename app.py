"""Aplikasi demo klasifikasi pakaian Fashion-MNIST (Gradio).

Memakai dua berkas dari folder `models/`:
  - model_final.tflite : prediksi cepat
  - best_model.keras   : penjelasan prediksi dengan Grad-CAM
  - model_info.json    : nama kelas, nama lapisan Grad-CAM, ringkasan model

Praproses gambar pengguna harus identik dengan fungsi `preprocess_user_image` di notebook.
"""
import json
import os
import threading

import gradio as gr
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

with open(os.path.join(MODEL_DIR, "model_info.json"), encoding="utf-8") as f:
    INFO = json.load(f)
CLASS_NAMES = INFO["class_names"]
GRADCAM_LAYER = INFO["gradcam_layer"]

# ---- Model -----------------------------------------------------------------
keras_model = keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))

try:   # LiteRT menggantikan tf.lite.Interpreter pada versi TensorFlow yang lebih baru
    from ai_edge_litert.interpreter import Interpreter as _TFLiteInterpreter
except Exception:
    _TFLiteInterpreter = tf.lite.Interpreter

_interp = _TFLiteInterpreter(model_path=os.path.join(MODEL_DIR, "model_final.tflite"))
_interp.allocate_tensors()
_in = _interp.get_input_details()[0]
_out = _interp.get_output_details()[0]
_lock = threading.Lock()   # interpreter TFLite tidak aman dipakai paralel


def _build_grad_model(model, layer_name):
    """Rangkai ulang graf agar keluaran lapisan konvolusi terakhir ikut menjadi keluaran."""
    inp = keras.Input(shape=model.input_shape[1:])
    x, conv_out = inp, None
    for layer in model.layers:
        if isinstance(layer, layers.InputLayer):
            continue
        x = layer(x)
        if layer.name == layer_name:
            conv_out = x
    return keras.Model(inp, [conv_out, x])


grad_model = _build_grad_model(keras_model, GRADCAM_LAYER)


# ---- Praproses -------------------------------------------------------------
def preprocess_user_image(pil_img):
    """Foto/gambar bebas -> array (1, 28, 28, 1) float32 skala 0-255."""
    img = ImageOps.exif_transpose(pil_img)        # hormati orientasi foto dari HP
    img = ImageOps.grayscale(img)
    if np.asarray(img).mean() > 127:              # latar terang -> inversi agar mirip Fashion-MNIST
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    side = max(img.size)
    img = ImageOps.pad(img, (side, side), color=0)    # jadikan persegi tanpa mengubah rasio
    img = img.resize((28, 28), Image.LANCZOS)
    return np.asarray(img, dtype="float32")[np.newaxis, ..., np.newaxis]


# ---- Prediksi dan Grad-CAM -------------------------------------------------
def tflite_probs(x):
    with _lock:
        _interp.set_tensor(_in["index"], x)
        _interp.invoke()
        return _interp.get_tensor(_out["index"])[0].copy()


def gradcam(x, class_idx):
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


def overlay_image(img28, cam, size=280):
    base = np.repeat(img28[:, :, 0:1], 3, axis=2) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    blend = np.clip(0.55 * base + 0.45 * heat, 0, 1)
    return Image.fromarray((blend * 255).astype("uint8")).resize((size, size), Image.BICUBIC)


def predict(pil_img):
    if pil_img is None:
        return None, None, None, "Silakan unggah foto pakaian terlebih dahulu."
    x = preprocess_user_image(pil_img)
    probs = tflite_probs(x)
    top = int(probs.argmax())

    labels = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    seen = Image.fromarray(x[0, :, :, 0].astype("uint8")).resize((280, 280), Image.NEAREST)
    cam_img = overlay_image(x[0], gradcam(x, top))

    note = f"Prediksi: {CLASS_NAMES[top]} ({probs[top] * 100:.1f}%)."
    if probs[top] < 0.6:
        note += (" Model kurang yakin. Coba foto satu pakaian dengan latar polos dan kontras jelas "
                 "(model dilatih pada citra Fashion-MNIST 28x28 grayscale).")
    return labels, seen, cam_img, note


# ---- Antarmuka -------------------------------------------------------------
DESCRIPTION = (
    "Unggah foto satu pakaian atau alas kaki (latar polos paling baik). Model CNN mengklasifikasikannya "
    "ke 10 kelas Fashion-MNIST. Gambar tengah menunjukkan masukan yang dilihat model (28x28 grayscale), "
    "gambar kanan adalah peta panas Grad-CAM yang menandai area paling berpengaruh pada keputusan model."
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Unggah foto pakaian"),
    outputs=[
        gr.Label(num_top_classes=3, label="Hasil klasifikasi"),
        gr.Image(label="Masukan model (28x28)"),
        gr.Image(label="Grad-CAM"),
        gr.Textbox(label="Catatan"),
    ],
    title="Klasifikasi Citra Pakaian Fashion-MNIST (CNN)",
    description=DESCRIPTION,
)

if __name__ == "__main__":
    demo.launch()
