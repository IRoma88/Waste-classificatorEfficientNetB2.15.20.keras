import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Cachear modelo para que no se recargue cada vez
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("EfficientNetB2_finetuned.keras", compile=False)
    return model

model = load_model()

# Etiquetas entrenadas (en el mismo orden que tu dataset)
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

st.title("♻️ Clasificador de Residuos con EfficientNetB2")
st.markdown("Sube una imagen para predecir su tipo de residuo.")

uploaded_file = st.file_uploader("📸 Sube una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen subida
    img = image.load_img(uploaded_file, target_size=(380, 380))
    st.image(img, caption="Imagen subida", use_container_width=True)

    # Preprocesar imagen
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización

    # Predicción
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_label = CLASS_NAMES[pred_index]
    confidence = np.max(prediction) * 100

    st.success(f"✅ Predicción: **{pred_label}** ({confidence:.2f}% de confianza)")
