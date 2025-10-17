import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuración de la página
st.set_page_config(page_title="♻️ Waste Classificator", layout="centered")
st.title("♻️ Waste Classificator - EfficientNetB2 (Stable)")

# --- Ruta al modelo .keras simple ---
MODEL_PATH = os.path.join("models", "EfficientNetB2_simple.keras")

# --- Clases del dataset ---
class_names = [
    "BLUECardboardBriks", "BLUEGlassBottles1", "BLUEGlassBottles2",
    "BLUEMetalDrinksTupper", "BLUEPaperBook", "BLUEPlastics1", "BLUEPlastics2",
    "BrownOrganico", "GRAYThrash", "SPECIALDropOff", "SPECIALHHW",
    "SPECIALMedicalOff", "SPECIALTakeBackShop"
]

# --- Subida de imagen ---
uploaded_file = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrir y mostrar imagen
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    # Preprocesamiento
    IMG_SIZE = (224, 224)  # tamaño más pequeño para ahorrar memoria
    img = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # --- Cargar modelo solo al momento de la predicción ---
    with st.spinner("Cargando modelo..."):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Modelo cargado")

    # --- Predicción ---
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.markdown(f"### 🧠 Predicción: **{pred_class}** ({confidence:.2f}% de confianza)")
