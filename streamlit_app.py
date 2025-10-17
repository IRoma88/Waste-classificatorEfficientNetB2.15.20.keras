# streamlit_app.py - VERSIÓN DEFINITIVA
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

# Configuración
st.set_page_config(page_title="Clasificador de Residuos", page_icon="♻️", layout="centered")
st.title("♻️ Clasificador de Residuos")

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/EfficientNetB2.15.20.keras"
IMG_SIZE = (380, 380)  # EfficientNetB2 espera 380x380

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Archivo no encontrado: {MODEL_PATH}")
            return None
        
        with st.spinner("🔄 Cargando modelo..."):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        st.success("✅ Modelo cargado exitosamente")
        st.info(f"📊 Forma de entrada esperada: {model.input_shape}")
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        return None

# Cargar modelo
model = load_model()

# --- FUNCIONES DE PREPROCESAMIENTO MEJORADAS ---
def force_3_channels(image_array):
    """Fuerza la imagen a tener exactamente 3 canales RGB"""
    # Debug: mostrar forma original
    st.write(f"🔍 Forma original de la imagen: {image_array.shape}")
    
    if len(image_array.shape) == 2:
        # Escala de grises -> RGB
        st.info("🔄 Convirtiendo escala de grises a RGB")
        return np.stack([image_array] * 3, axis=-1)
    
    elif image_array.shape[2] == 1:
        # 1 canal -> RGB
        st.info("🔄 Convirtiendo 1 canal a RGB")
        return np.concatenate([image_array] * 3, axis=-1)
    
    elif image_array.shape[2] == 4:
        # RGBA -> RGB (con fondo blanco)
        st.info("🔄 Convirtiendo RGBA a RGB")
        rgb = image_array[:, :, :3]
        alpha = image_array[:, :, 3:4] / 255.0
        white_bg = np.ones_like(rgb) * 255
        result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        return result
    
    elif image_array.shape[2] == 3:
        # Ya es RGB
        return image_array
    
    else:
        # Formato desconocido, tomar primeros 3 canales
        st.warning(f"⚠️ Formato inesperado: {image_array.shape}, tomando primeros 3 canales")
        return image_array[:, :, :3]

def preprocess_image(uploaded_file):
    """Preprocesamiento robusto que garantiza 3 canales"""
    try:
        # Leer imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            st.error("❌ No se pudo decodificar la imagen")
            return None, None
        
        st.write(f"📐 Forma después de cv2.imdecode: {img.shape}")
        
        # OpenCV lee en BGR, convertir a RGB
        if len(img.shape) == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Forzar 3 canales
        img_rgb = force_3_channels(img)
        st.write(f"🎯 Forma después de forzar RGB: {img_rgb.shape}")
        
        # Redimensionar
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        st.write(f"📏 Forma después de redimensionar: {img_resized.shape}")
        
        # Convertir a float32 y normalizar
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Verificar forma final
        if img_float.shape != (*IMG_SIZE, 3):
            st.error(f"❌ Forma final incorrecta: {img_float.shape}. Esperado: {(*IMG_SIZE, 3)}")
            return None, None
        
        # Añadir dimensión del batch
        img_batch = np.expand_dims(img_float, axis=0)
        st.success(f"✅ Preprocesamiento completado. Forma final: {img_batch.shape}")
        
        # Crear imagen para mostrar
        img_display = Image.fromarray(img_resized.astype(np.uint8))
        
        return img_batch, img_display
        
    except Exception as e:
        st.error(f"❌ Error en preprocesamiento: {str(e)}")
        return None, None

def predict(img_array):
    """Realiza predicción"""
    try:
        # Verificar forma
        if img_array.shape != (1, 380, 380, 3):
            st.error(f"❌ Forma incorrecta para predicción: {img_array.shape}")
            return "Error: Forma de imagen incorrecta", 0.0
        
        # Predicción
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return CLASS_NAMES[class_idx], confidence
        
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL ---
if model is not None:
    st.success("✅ ¡Sistema listo para clasificar!")
    
    # Información del sistema
    with st.expander("📊 Información Técnica"):
        st.write(f"**Modelo:** EfficientNetB2")
        st.write(f"**Entrada esperada:** {model.input_shape}")
        st.write(f"**Tamaño imagen:** {IMG_SIZE}")
        st.write(f"**Clases:** {len(CLASS_NAMES)}")
    
    # Subir imagen
    uploaded_file = st.file_uploader(
        "Sube una imagen de residuo", 
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    
    if uploaded_file is not None:
        # Preprocesar
        img_array, img_display = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Mostrar resultados
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_display, caption="Imagen procesada", use_column_width=True)
                st.write(f"**Archivo:** {uploaded_file.name}")
            
            with col2:
                # Predecir
                with st.spinner("🔍 Clasificando..."):
                    class_name, confidence = predict(img_array)
                
                if "Error" not in class_name:
                    # Resultados
                    st.success(f"**🎯 Resultado:** {class_name}")
                    st.progress(confidence)
                    st.write(f"**Confianza:** {confidence:.1%}")
                    
                    # Información de categoría
                    st.markdown("---")
                    if "BlueRecyclable" in class_name:
                        st.info("🔵 **RECICLABLE** - Contenedor Azul")
                    elif "BrownCompost" in class_name:
                        st.info("🟤 **ORGÁNICO** - Contenedor Marrón")
                    elif "GrayTrash" in class_name:
                        st.info("⚪ **RESTO** - Contenedor Gris")
                    else:
                        st.warning("🟡 **ESPECIAL** - Consulta normas locales")
                else:
                    st.error(f"❌ {class_name}")

else:
    st.error("🚫 Sistema no disponible")

# Footer
st.markdown("---")
st.caption("♻️ Clasificador de Residuos | EfficientNetB2")
