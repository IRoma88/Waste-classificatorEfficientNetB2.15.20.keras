import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# === CONFIGURACIÓN ===
MODEL_PATH = os.path.join("models", "EfficientNetB2.15.20.keras")

CLASS_NAMES = [
    "BLUECardboardBriks",
    "BLUEGlassBottles1",
    "BLUEGlassBottles2",
    "BLUEMetalDrinksTupper",
    "BLUEPaperBook",
    "BLUEPlastics1",
    "BLUEPlastics2",
    "BrownOrganico",
    "GRAYThrash",
    "SPECIALDropOff",
    "SPECIALHHW",
    "SPECIALMedicalOff",
    "SPECIALTakeBackShop"
]

IMG_SIZE = (380, 380)  # tamaño usado al entrenar

# === CARGAR MODELO CON CACHE ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# === INTERFAZ ===
st.title("♻️ Clasificador de Residuos con EfficientNetB2")
st.markdown(
    """
    Este modelo clasifica imágenes en 13 categorías de residuos:
    **Blue (reciclables)**, **Gray (trash)**, **Brown (orgánico)** y **Special (HHW)**.
    """
)

uploaded_file = st.file_uploader("📸 Sube una imagen del residuo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    st.image(img, caption="Imagen subida", use_container_width=True)

    # Preprocesamiento
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)
    pred_idx = np.argmax(prediction)
    pred_label = CLASS_NAMES[pred_idx]
    confidence = np.max(prediction) * 100

    # Mostrar resultado
    st.success(f"**Predicción:** {pred_label} ({confidence:.2f}% de confianza)")
