# streamlit_app.py - VERSIÓN CORREGIDA
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Configuración de página
st.set_page_config(
    page_title="Clasificador de Residuos / Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Clasificador de Residuos / Waste Classifier")
st.write("Sube una imagen de residuo para clasificarlo / Upload a waste image to classify it")

# --- CONFIGURACIÓN CORREGIDA ---
MODEL_PATH = "models/EfficientNetB2_epochs15-20.keras"
# ⚠️ USA SOLO EL FILE ID - NO EL ENLACE COMPLETO
FILE_ID = "1PcSynIU3Od_82zdHOerJRx3NLyEYbAUH"
IMG_SIZE = (380, 380)

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

# --- CARGA DEL MODELO CORREGIDA ---
@st.cache_resource
def download_and_load_model():
    # Crear carpeta models
    os.makedirs("models", exist_ok=True)
    
    # Si ya existe el modelo, cargarlo
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("✅ Modelo cargado desde caché / Model loaded from cache")
            return model
        except Exception as e:
            st.warning(f"⚠️ Modelo corrupto, reintentando... / Corrupt model, retrying...")
            os.remove(MODEL_PATH)
    
    # Descargar modelo
    st.info("📥 Descargando modelo... / Downloading model...")
    
    try:
        # ⚠️ FORMATO CORRECTO para gdown
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        
        # Verificar que se descargó
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("✅ Modelo descargado y cargado / Model downloaded and loaded")
            return model
        else:
            st.error("❌ El archivo se descargó vacío / File downloaded empty")
            return None
            
    except Exception as e:
        st.error(f"""
        ❌ Error descargando modelo / Error downloading model: {e}
        
        **🔧 SOLUCIÓN / SOLUTION:**
        1. **Verifica que el archivo sea PÚBLICO** en Google Drive
        2. **Haz clic derecho** → **Compartir** → **"Cualquier persona con el enlace"**
        3. **Asegúrate** de que diga "Cualquier persona con el enlace" como "Lector"
        4. **Recarga** esta aplicación / **Reload** this app
        
        **📋 Pasos detallados / Detailed steps:**
        - Ve a https://drive.google.com
        - Encuentra el archivo 'EfficientNetB2_epochs15-20.keras'
        - Clic derecho → Compartir → Cambiar a "Cualquier persona con el enlace"
        - Guarda los cambios
        - Vuelve aquí y recarga
        """)
        return None

# Cargar modelo
model = download_and_load_model()

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
        st.error(f"❌ Error procesando imagen / Error processing image: {e}")
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
    
    uploaded_file = st.file_uploader(
        "Sube una imagen / Upload an image", 
        type=["jpg", "jpeg", "png", "webp"]
    )
    
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
