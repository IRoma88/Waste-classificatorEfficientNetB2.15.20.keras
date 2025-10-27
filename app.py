import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Configuración de la página ---
st.set_page_config(
    page_title="♻️ Waste Classificator / Clasificador de Residuos",
    layout="centered"
)

st.title("♻️ Waste Classificator / Clasificador de Residuos - EfficientNetB2")

# --- Ruta al modelo SavedModel ---
MODEL_PATH = os.path.join("models", "EfficientNetB2_savedmodel")

# --- Clases del dataset ---
class_names = [
    "BLUECardboardBriks", "BLUEGlassBottles1", "BLUEGlassBottles2",
    "BLUEMetalDrinksTupper", "BLUEPaperBook", "BLUEPlastics1", "BLUEPlastics2",
    "BrownOrganico", "GRAYThrash", "SPECIALDropOff", "SPECIALHHW",
    "SPECIALMedicalOff", "SPECIALTakeBackShop"
]

# --- Nombres legibles para la app (mismo orden que las etiquetas internas) ---
display_labels = [
    "♻️ Blue - Cardboard & Briks",
    "♻️ Blue - Glass Bottles (Type 1)",
    "♻️ Blue - Glass Bottles (Type 2)",
    "♻️ Blue - Metal Cans & Tupperware",
    "♻️ Blue - Paper & Books",
    "♻️ Blue - Plastics (Type 1)",
    "♻️ Blue - Plastics (Type 2)",
    "🍃 Brown - Organic Waste",
    "🗑️ Gray - General Trash",
    "🏪 Take-Back Shop Items",
    "⚠️ Hazardous Household Waste (HHW)",
    "🏥 Medical Waste",
    "📦 Drop-off Point Items"
]

# --- Precargar modelo (solo una vez) ---
@st.cache_resource
def load_model():
    return tf.saved_model.load(MODEL_PATH)

model = load_model()

st.success("✅ Modelo cargado / Model loaded")

# --- Subida de imagen / Image upload ---
uploaded_file = st.file_uploader(
    "Sube una imagen para clasificar / Upload an image to classify",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Abrir y mostrar la imagen / Open and display the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen subida / Uploaded image", use_container_width=True)

    # Preprocesamiento / Preprocessing
    IMG_SIZE = (380, 380)
    img = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img), axis=0).astype(np.float32)
    img_array = preprocess_input(img_array)

    # --- Predicción / Prediction ---
    with st.spinner("Prediciendo... / Predicting..."):
        infer = model.signatures["serving_default"]
        preds_dict = infer(tf.constant(img_array))
        preds = list(preds_dict.values())[0].numpy()

        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

    st.markdown(f"### 🧠 Predicción / Prediction: **{pred_class}** ({confidence:.2f}% de confianza / confidence)")
