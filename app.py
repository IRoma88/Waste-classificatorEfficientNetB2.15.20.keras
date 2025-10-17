import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# Configuración de la página
st.set_page_config(page_title="♻️ Waste Classificator", layout="centered")
st.title("♻️ Waste Classificator - EfficientNetB2 (Optimized)")

# Ruta al modelo SavedModel
MODEL_PATH = os.path.join("models", "EfficientNetB2_savedmodel")

# --- Clases del dataset ---
class_names = [
    "BLUECardboardBriks", "BLUEGlassBottles1", "BLUEGlassBottles2",
    "BLUEMetalDrinksTupper", "BLUEPaperBook", "BLUEPlastics1", "BLUEPlastics2",
    "BrownOrganico", "GRAYThrash", "SPECIALDropOff", "SPECIALHHW",
    "SPECIALMedicalOff", "SPECIALTakeBackShop"
]

# --- Subida de imagen ---
uploaded_file = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Abrir y mostrar la imagen
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    # Preprocesamiento exacto de EfficientNetB2
    IMG_SIZE = (380, 380)  # tamaño usado en tu entrenamiento
    img = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img), axis=0).astype(np.float32)
    img_array = preprocess_input(img_array)  # normalización específica EfficientNet

    # --- Cargar modelo SavedModel ---
    with st.spinner("Cargando modelo..."):
        model = tf.saved_model.load(MODEL_PATH)
    st.success("✅ Modelo cargado")

    # --- Predicción ---
    infer = model.signatures["serving_default"]
    preds_dict = infer(tf.constant(img_array))
    preds = list(preds_dict.values())[0].numpy()  # convertir dict a array

    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)*100

    st.markdown(f"### 🧠 Predicción: **{pred_class}** ({confidence:.2f}% de confianza)")
