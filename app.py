import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.set_page_config(page_title="♻️ Waste Classificator", layout="centered")
st.title("♻️ Waste Classificator - EfficientNetB2")

# --- Carpeta para modelos ---
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Ruta y URL del modelo en Google Drive ---
MODEL_PATH = os.path.join(MODEL_DIR, "EfficientNetB2_repaired.keras")
# Sustituye <FILE_ID> por el ID de tu archivo en Drive
# Puedes obtenerlo de la URL compartida: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
DRIVE_URL = "https://drive.google.com/uc?id=<FILE_ID>"

# --- Descargar modelo si no existe ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("Descargando modelo desde Google Drive..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# --- Cargar modelo ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

with st.spinner("Cargando modelo..."):
    model = load_model()
st.success("Modelo cargado correctamente ✅")

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
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen subida", use_column_width=True)

    # --- Preprocesamiento ---
    IMG_SIZE = (380, 380)  # tamaño usado en tu entrenamiento
    img = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # --- Predicción ---
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.markdown(f"### 🧠 Predicción: **{pred_class}** ({confidence:.2f}% de confianza)")
