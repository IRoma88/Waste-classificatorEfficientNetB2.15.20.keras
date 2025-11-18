import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Configuración de la página ---
st.set_page_config(
    page_title="♻️ Waste Classificator / Clasificador de Residuos",
    layout="centered"
)

st.title("♻️ Waste Classificator / Clasificador de Residuos - EfficientNetB2")


# -------------------------------------------------------------
# 🗂️ 1. TABLA DE COLORES DE CONTENEDORES
# -------------------------------------------------------------
st.markdown("## 🗂️ Bin Types & Color Codes / Tipos de Contenedores y Colores")

bin_table = {
    "Bin Type / Tipo de Contenedor": [
        "Blue – Paper, Plastics, Glass, Metals",
        "Brown – Organic Waste",
        "Gray – General Trash",
        "Special – Drop-off Points",
        "Hazardous Household Waste (HHW)",
        "Medical Waste",
        "Take-Back Shop"
    ],
    "Color / Color": [
        "🔵 Blue",
        "🟤 Brown",
        "⚪ Gray",
        "🟧 Orange",
        "⚠️ Yellow",
        "❤️ Red",
        "🏪 Various"
    ]
}

st.table(pd.DataFrame(bin_table))


# -------------------------------------------------------------
# 🔥 2. CONFIGURACIÓN DEL MODELO
# -------------------------------------------------------------
MODEL_PATH = os.path.join("models", "EfficientNetB2_savedmodel")

internal_labels = [
    "BLUECardboardBriks", "BLUEGlassBottles1", "BLUEGlassBottles2",
    "BLUEMetalDrinksTupper", "BLUEPaperBook", "BLUEPlastics1", "BLUEPlastics2",
    "BrownOrganico", "GRAYThrash", "SPECIALDropOff", "SPECIALHHW",
    "SPECIALMedicalOff", "SPECIALTakeBackShop"
]

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

# -------------------------------------------------------------
# 📝 3. DESCRIPCIONES AUTOMÁTICAS POR CATEGORÍA
# -------------------------------------------------------------
descriptions = {
    "♻️ Blue - Cardboard & Briks": 
        "Use the Blue bin. Clean and flatten boxes if possible.",
    "♻️ Blue - Glass Bottles (Type 1)": 
        "Use the Blue bin. Remove caps before recycling.",
    "♻️ Blue - Glass Bottles (Type 2)": 
        "Blue bin. Do not include broken glass.",
    "♻️ Blue - Metal Cans & Tupperware": 
        "Blue bin. Clean cans and avoid mixing with organic waste.",
    "♻️ Blue - Paper & Books": 
        "Blue bin. Avoid paper contaminated with food.",
    "♻️ Blue - Plastics (Type 1)": 
        "Blue bin. Rinse plastics before disposal.",
    "♻️ Blue - Plastics (Type 2)": 
        "Blue bin. Keep plastic bags separate.",
    "🍃 Brown - Organic Waste": 
        "Brown bin. Includes food scraps and compostables.",
    "🗑️ Gray - General Trash": 
        "Gray bin. Items that cannot be recycled.",
    "🏪 Take-Back Shop Items": 
        "Take-back shop. Reusable items can be donated.",
    "⚠️ Hazardous Household Waste (HHW)": 
        "HHW facility. Never throw into regular bins.",
    "🏥 Medical Waste": 
        "Medical waste must go to authorized collection points.",
    "📦 Drop-off Point Items": 
        "Drop-off centers handle electronics, batteries, etc."
}


# -------------------------------------------------------------
# ⚙️ 4. CARGAR EL MODELO
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.saved_model.load(MODEL_PATH)

model = load_model()
st.success("✅ Modelo cargado correctamente / Model loaded successfully")


# -------------------------------------------------------------
# 📤 5. SUBIDA Y CLASIFICACIÓN DE IMÁGENES
# -------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Sube una imagen para clasificar / Upload an image to classify",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Imagen subida / Uploaded image", use_container_width=True)

    # Preprocesamiento
    IMG_SIZE = (380, 380)
    img = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img), axis=0).astype(np.float32)
    img_array = preprocess_input(img_array)

    with st.spinner("🔍 Analizando imagen... / Analyzing image..."):
        infer = model.signatures["serving_default"]
        preds_dict = infer(tf.constant(img_array))
        preds = list(preds_dict.values())[0].numpy()

        pred_index = np.argmax(preds)
        pred_class = display_labels[pred_index]
        confidence = np.max(preds) * 100

    st.markdown(f"## 🧠 Predicción / Prediction: **{pred_class}**")
    st.progress(float(confidence) / 100)
    st.write(f"**Confianza / Confidence:** {confidence:.2f}%")

    # Descripción automática
    st.markdown("### 📝 Description / Descripción:")
    st.info(descriptions.get(pred_class, "No description available."))
