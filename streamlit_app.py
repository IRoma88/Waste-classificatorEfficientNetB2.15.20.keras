# streamlit_app.py - VERSIÓN CORREGIDA CON DETECCIÓN AUTOMÁTICA
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

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

# --- CARGA DEL MODELO CON DETECCIÓN AUTOMÁTICA ---
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Archivo no encontrado: {MODEL_PATH}")
            return None, None
        
        with st.spinner("🔄 Cargando modelo..."):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # OBTENER LA FORMA DE ENTRADA DEL MODELO
        input_shape = model.input_shape
        if input_shape is None or len(input_shape) != 4:
            st.error("❌ No se pudo determinar la forma de entrada del modelo")
            return None, None
            
        # Extraer tamaño de imagen (altura, ancho)
        img_height, img_width = input_shape[1], input_shape[2]
        st.success(f"✅ Modelo cargado exitosamente")
        st.info(f"📊 Forma de entrada detectada: {input_shape}")
        st.info(f"🎯 Tamaño de imagen requerido: {img_height}x{img_width}")
        
        return model, (img_height, img_width)
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        return None, None

# Cargar modelo y tamaño
model, IMG_SIZE = load_model()

# --- FUNCIONES DE PREPROCESAMIENTO ---
def debug_image_shape(image_array, step_name):
    """Función para debug de formas de imagen"""
    st.write(f"🔍 {step_name}: {image_array.shape}")

def force_3_channels(image_array):
    """Fuerza la imagen a tener exactamente 3 canales RGB"""
    debug_image_shape(image_array, "Forma original")
    
    if len(image_array.shape) == 2:
        # Escala de grises -> RGB
        st.info("🔄 Convirtiendo escala de grises a RGB")
        result = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 1:
        # 1 canal -> RGB
        st.info("🔄 Convirtiendo 1 canal a RGB")
        result = np.concatenate([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        # RGBA -> RGB
        st.info("🔄 Convirtiendo RGBA a RGB")
        result = image_array[:, :, :3]
    elif image_array.shape[2] == 3:
        # Ya es RGB
        result = image_array
    else:
        # Formato desconocido
        st.warning("⚠️ Formato desconocido, tomando primeros 3 canales")
        result = image_array[:, :, :3]
    
    debug_image_shape(result, "Después de conversión RGB")
    return result

def preprocess_image(uploaded_file):
    """Preprocesamiento que garantiza la forma correcta"""
    try:
        if IMG_SIZE is None:
            st.error("❌ No se detectó el tamaño de imagen requerido")
            return None, None
            
        # Reiniciar el puntero del archivo
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Leer imagen con OpenCV
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if img is None:
            st.error("❌ No se pudo decodificar la imagen")
            return None, None
        
        debug_image_shape(img, "Después de cv2.imdecode")
        
        # Si la imagen es BGR (formato OpenCV), convertir a RGB
        if len(img.shape) == 3 and img.shape[2] >= 3:
            # Verificar si es BGR (OpenCV)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Forzar 3 canales RGB
        img_rgb = force_3_channels(img)
        
        # Redimensionar al tamaño EXACTO que espera el modelo
        img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]))  # (width, height)
        debug_image_shape(img_resized, "Después de redimensionar")
        
        # Verificar canales
        if img_resized.shape[2] != 3:
            st.error(f"❌ Error: imagen tiene {img_resized.shape[2]} canales, se requieren 3")
            return None, None
        
        # Convertir a float32 y normalizar
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Verificar forma final
        expected_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
        if img_float.shape != expected_shape:
            st.error(f"❌ Forma final incorrecta: {img_float.shape}. Esperado: {expected_shape}")
            return None, None
        
        # Añadir dimensión del batch
        img_batch = np.expand_dims(img_float, axis=0)
        debug_image_shape(img_batch, "Forma final para el modelo")
        
        # Crear imagen para mostrar
        img_display = Image.fromarray((img_resized).astype(np.uint8))
        
        st.success("✅ Preprocesamiento completado correctamente")
        return img_batch, img_display
        
    except Exception as e:
        st.error(f"❌ Error en preprocesamiento: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

def predict(img_array):
    """Realiza predicción con verificación de forma"""
    try:
        # Verificar forma exacta
        expected_shape = (1, IMG_SIZE[0], IMG_SIZE[1], 3)
        if img_array.shape != expected_shape:
            st.error(f"❌ Forma incorrecta para predicción: {img_array.shape}. Esperado: {expected_shape}")
            return "Error: Forma de imagen incorrecta", 0.0
        
        # Realizar predicción
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return CLASS_NAMES[class_idx], confidence
        
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL ---
if model is not None and IMG_SIZE is not None:
    st.success(f"✅ ¡Sistema listo! Tamaño requerido: {IMG_SIZE[0]}x{IMG_SIZE[1]}px")
    
    # Información del sistema
    with st.expander("📊 Información Técnica"):
        st.write(f"**Modelo:** EfficientNetB2")
        st.write(f"**Entrada esperada:** {model.input_shape}")
        st.write(f"**Tamaño imagen:** {IMG_SIZE[0]}x{IMG_SIZE[1]}")
        st.write(f"**Canales:** 3 (RGB)")
        st.write(f"**Clases:** {len(CLASS_NAMES)}")
    
    # Subir imagen
    uploaded_file = st.file_uploader(
        f"Sube una imagen de residuo (se redimensionará a {IMG_SIZE[0]}x{IMG_SIZE[1]}px)", 
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    
    if uploaded_file is not None:
        # Preprocesar
        with st.spinner("🔄 Procesando imagen..."):
            img_array, img_display = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Mostrar en columnas
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_display, caption=f"Imagen procesada {IMG_SIZE[0]}x{IMG_SIZE[1]}", use_column_width=True)
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
                        material = class_name.split("_")[1]
                        st.info(f"🔵 **RECICLABLE** - Contenedor Azul\n\nMaterial: {material}")
                    elif "BrownCompost" in class_name:
                        st.info("🟤 **ORGÁNICO** - Contenedor Marrón\n\nRestos de comida, frutas, verduras")
                    elif "GrayTrash" in class_name:
                        st.info("⚪ **RESTO** - Contenedor Gris\n\nMateriales no reciclables")
                    else:
                        st.warning("🟡 **CATEGORÍA ESPECIAL** - Consulta normas locales")
                        
                    # Interpretación de confianza
                    if confidence > 0.8:
                        st.success("🟢 **ALTA CONFIANZA**")
                    elif confidence > 0.6:
                        st.info("🟡 **CONFIANZA MEDIA**")
                    else:
                        st.warning("🔴 **BAJA CONFIANZA** - Verificar manualmente")
                else:
                    st.error(f"❌ {class_name}")

else:
    st.error("🚫 Sistema no disponible")
    
    # Información de solución de problemas
    with st.expander("🔧 Solución de Problemas Detallada"):
        st.write("""
        **Problema detectado:** Incompatibilidad entre la forma de entrada del modelo y la imagen procesada
        
        **Posibles causas:**
        1. El modelo fue entrenado con un tamaño diferente al esperado
        2. Problema de versión de TensorFlow
        3. Archivo de modelo corrupto o incompatible
        
        **Soluciones a intentar:**
        - Verifica que el archivo del modelo sea correcto
        - Prueba con diferentes imágenes
        - Revisa los logs para más detalles del error
        """)

# Footer
st.markdown("---")
st.caption("♻️ Clasificador de Residuos | EfficientNetB2")
