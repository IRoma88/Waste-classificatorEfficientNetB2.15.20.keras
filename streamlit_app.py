# streamlit_app.py - VERSIÓN CORREGIDA PARA STREAMLIT CLOUD
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuración de página
st.set_page_config(
    page_title="Clasificador de Residuos / Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Clasificador de Residuos / Waste Classifier")
st.write("Sube una imagen de residuo para clasificarlo / Upload a waste image to classify it")

# --- CONFIGURACIÓN CORREGIDA ---
MODEL_PATH = "models/EfficientNetB2.15.20.keras"  # ⚠️ ARCHIVO LOCAL, NO URL
IMG_SIZE = (380, 380)

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

# --- CARGA SIMPLE DEL MODELO LOCAL ---
@st.cache_resource
def load_model():
    try:
        # Verificar que el archivo existe LOCALMENTE
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Modelo no encontrado en: {MODEL_PATH}")
            st.info("💡 El modelo debe estar en la carpeta 'models/' del repositorio")
            return None
        
        # Cargar el modelo LOCAL
        with st.spinner("🔄 Cargando modelo..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        
        st.success("✅ Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {e}")
        return None

# Cargar modelo
model = load_model()

# --- FUNCIONES ---
def preprocess_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {e}")
        return None, None

def predict(img_array):
    try:
        preds = model.predict(img_array, verbose=0)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL ---
if model is not None:
    st.success("✅ ¡Listo para clasificar! / Ready to classify!")
    
    with st.expander("📊 Información del Modelo / Model Information"):
        st.write(f"**Arquitectura:** EfficientNetB2")
        st.write(f"**Clases:** {len(CLASS_NAMES)}")
        st.write(f"**Tamaño entrada:** {IMG_SIZE}")
        st.write("**Ubicación:** Archivo local en el repositorio")
    
    uploaded_file = st.file_uploader("Sube una imagen / Upload an image", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        img_array, img_display = preprocess_image(uploaded_file)
        if img_array is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img_display, use_column_width=True)
            with col2:
                with st.spinner("🔍 Clasificando... / Classifying..."):
                    pred_class, confidence = predict(img_array)
                if "Error" not in pred_class:
                    st.success(f"**🎯 Predicción:** {pred_class}")
                    st.progress(confidence)
                    st.write(f"**📊 Confianza:** {confidence*100:.2f}%")
                    
                    if "BlueRecyclable" in pred_class:
                        st.info("🔵 **Contenedor Azul - Reciclable**")
                    elif "BrownCompost" in pred_class:
                        st.info("🟤 **Contenedor Marrón - Orgánico**")
                    elif "GrayTrash" in pred_class:
                        st.info("⚪ **Contenedor Gris - Resto**")
                    elif "SPECIAL" in pred_class:
                        st.warning("🟡 **Categoría Especial**")
                else:
                    st.error(pred_class)
else:
    st.error("🚫 No se pudo cargar el modelo / Could not load model")

st.markdown("---")
st.caption("♻️ Clasificador de Residuos | Waste Classifier")
