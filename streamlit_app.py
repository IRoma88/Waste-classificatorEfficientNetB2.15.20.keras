# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuración de página
st.set_page_config(
    page_title="Clasificador de Residuos",
    page_icon="♻️",
    layout="centered"
)

# Título bilingüe
st.title("♻️ Clasificador de Residuos / Waste Classifier")
st.write("Sube una imagen de residuo para clasificarlo / Upload a waste image to classify it")

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/EfficientNetB2_epochs15-20.keras"
IMG_SIZE = (380, 380)

# Clases (AJUSTA SEGÚN TU ENTRENAMIENTO)
CLASS_NAMES = [
    "BlueRecyclable_Cardboard",
    "BlueRecyclable_Glass", 
    "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", 
    "BlueRecyclable_Plastics",
    "BrownCompost",
    "GrayTrash",
    "SPECIAL_DropOff",
    "SPECIAL_TakeBackShop",
    "SPECIAL_MedicalTakeBack",
    "SPECIAL_HHW"
]

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modelo no encontrado en: {MODEL_PATH}")
        st.info("💡 Asegúrate de que el archivo del modelo esté en la carpeta 'models/'")
        return None
    
    try:
        with st.spinner("🔄 Cargando modelo..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {e}")
        return None

# Cargar modelo al inicio
model = load_model()

# --- FUNCIONES ---
def preprocess_image(uploaded_file):
    """Preprocesa la imagen para el modelo"""
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
    """Realiza predicción con el modelo"""
    try:
        preds = model.predict(img_array, verbose=0)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL ---
if model is not None:
    st.success("✅ Modelo cargado - ¡Listo para clasificar!")
    
    # Información del modelo
    with st.expander("📊 Información del Modelo"):
        st.write(f"**Arquitectura:** EfficientNetB2")
        st.write(f"**Épocas de entrenamiento:** 15-20")
        st.write(f"**Tamaño de entrada:** 380x380 px")
        st.write(f"**Clases:** {len(CLASS_NAMES)} categorías")
    
    # Uploader de imagen
    uploaded_file = st.file_uploader(
        "Sube una imagen / Upload an image", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos soportados: JPG, JPEG, PNG, WEBP"
    )
    
    if uploaded_file is not None:
        # Procesar imagen
        img_array, img_display = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Mostrar imagen
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img_display, caption="Imagen subida / Uploaded image", use_column_width=True)
            
            with col2:
                # Realizar predicción
                with st.spinner("🔍 Clasificando..."):
                    pred_class, confidence = predict(img_array)
                
                if "Error" not in pred_class:
                    # Mostrar resultados
                    st.success(f"**🎯 Predicción / Prediction:** {pred_class}")
                    
                    # Barra de confianza
                    st.progress(confidence)
                    st.write(f"**📊 Confianza / Confidence:** {confidence*100:.2f}%")
                    
                    # Información de la categoría
                    st.markdown("---")
                    if "BlueRecyclable" in pred_class:
                        st.info("🔵 **Contenedor Azul - Reciclable / Blue Container - Recyclable**")
                    elif "BrownCompost" in pred_class:
                        st.info("🟤 **Contenedor Marrón - Orgánico / Brown Container - Organic**")
                    elif "GrayTrash" in pred_class:
                        st.info("⚪ **Contenedor Gris - Resto / Gray Container - General Waste**")
                    elif "SPECIAL" in pred_class:
                        st.warning("🟡 **Categoría Especial / Special Category**")
                else:
                    st.error(pred_class)

else:
    st.error("🚫 No se pudo cargar el modelo. Revisa la configuración.")

# Footer
st.markdown("---")
st.caption("Clasificador de Residuos con EfficientNetB2 | Waste Classifier with EfficientNetB2")
