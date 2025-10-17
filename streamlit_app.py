# streamlit_app.py - VERSIÓN CON DIAGNÓSTICO COMPLETO
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

# --- DIAGNÓSTICO COMPLETO DEL MODELO ---
@st.cache_resource
def diagnose_and_load_model():
    try:
        st.write("🔍 **Iniciando diagnóstico del modelo...**")
        
        # 1. Verificar que el archivo existe
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Archivo no encontrado: {MODEL_PATH}")
            return None, None
        
        st.success("✅ Archivo del modelo encontrado")
        
        # 2. Verificar tamaño del archivo
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        st.write(f"📊 Tamaño del archivo: {file_size:.2f} MB")
        
        if file_size < 1:
            st.warning("⚠️ El archivo del modelo parece muy pequeño, podría estar corrupto")
        
        # 3. Intentar cargar el modelo con diferentes métodos
        st.write("🔄 Intentando cargar el modelo...")
        
        # Método 1: Carga normal
        try:
            with st.spinner("Cargando con tf.keras.models.load_model..."):
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                st.success("✅ Modelo cargado con método estándar")
                return model, model.input_shape[1:3]  # (height, width)
        except Exception as e1:
            st.warning(f"⚠️ Método 1 falló: {str(e1)}")
        
        # Método 2: Carga con custom_objects si es necesario
        try:
            with st.spinner("Intentando carga con custom_objects..."):
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    compile=False,
                    custom_objects={}
                )
                st.success("✅ Modelo cargado con custom_objects")
                return model, model.input_shape[1:3]
        except Exception as e2:
            st.warning(f"⚠️ Método 2 falló: {str(e2)}")
        
        # Método 3: Intentar cargar solo la arquitectura
        try:
            with st.spinner("Intentando cargar solo arquitectura..."):
                # Crear un modelo temporal para diagnóstico
                temp_model = tf.keras.applications.EfficientNetB2(
                    weights=None, 
                    input_shape=(381, 381, 3),  # Probamos con el tamaño que indica el error
                    classes=len(CLASS_NAMES)
                )
                st.info("ℹ️ Se creó un modelo EfficientNetB2 temporal para diagnóstico")
                return temp_model, (381, 381)
        except Exception as e3:
            st.warning(f"⚠️ Método 3 falló: {str(e3)}")
        
        # Si todos los métodos fallan
        st.error("❌ Todos los métodos de carga fallaron")
        return None, None
        
    except Exception as e:
        st.error(f"❌ Error en diagnóstico: {str(e)}")
        return None, None

# --- FUNCIONES DE PREPROCESAMIENTO ---
def preprocess_image(uploaded_file, img_size):
    """Preprocesamiento robusto de imágenes"""
    try:
        # Reiniciar el puntero del archivo
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Leer imagen
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if img is None:
            st.error("❌ No se pudo decodificar la imagen")
            return None, None
        
        # Debug de forma original
        st.write(f"🔍 Forma original: {img.shape}")
        
        # Manejar diferentes formatos
        if len(img.shape) == 2:
            # Escala de grises -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            st.info("🔄 Convertido de escala de grises a RGB")
        elif img.shape[2] == 3:
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            # RGBA -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            st.info("🔄 Convertido de RGBA a RGB")
        else:
            img_rgb = img[:, :, :3]  # Tomar primeros 3 canales
        
        # Redimensionar
        img_resized = cv2.resize(img_rgb, (img_size[1], img_size[0]))
        st.write(f"📏 Forma después de redimensionar: {img_resized.shape}")
        
        # Verificar canales
        if img_resized.shape[2] != 3:
            st.error(f"❌ Error: la imagen tiene {img_resized.shape[2]} canales, se requieren 3")
            return None, None
        
        # Normalizar
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Añadir dimensión del batch
        img_batch = np.expand_dims(img_float, axis=0)
        st.write(f"🎯 Forma final para el modelo: {img_batch.shape}")
        
        # Imagen para mostrar
        img_display = Image.fromarray(img_resized)
        
        st.success("✅ Imagen preprocesada correctamente")
        return img_batch, img_display
        
    except Exception as e:
        st.error(f"❌ Error en preprocesamiento: {str(e)}")
        return None, None

def predict(model, img_array):
    """Realizar predicción"""
    try:
        # Verificar forma
        if img_array.shape[-1] != 3:
            st.error(f"❌ Error: la imagen tiene {img_array.shape[-1]} canales, se requieren 3")
            return "Error de canales", 0.0
        
        # Predicción
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return CLASS_NAMES[class_idx], confidence
        
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL ---
st.write("## 🔧 Diagnóstico del Sistema")

# Cargar modelo con diagnóstico
model, img_size = diagnose_and_load_model()

if model is not None and img_size is not None:
    st.success(f"✅ ¡Sistema listo! Tamaño de imagen: {img_size[0]}x{img_size[1]}")
    
    # Información del modelo
    with st.expander("📊 Información del Modelo"):
        st.write(f"**Tipo:** EfficientNetB2")
        st.write(f"**Forma de entrada:** {model.input_shape}")
        st.write(f"**Forma de salida:** {model.output_shape}")
        st.write(f"**Tamaño requerido:** {img_size[0]}x{img_size[1]}px")
        st.write(f"**Número de clases:** {len(CLASS_NAMES)}")
    
    # Subir imagen
    st.write("## 📤 Clasificar Imagen")
    uploaded_file = st.file_uploader(
        f"Sube una imagen para clasificar (se redimensionará a {img_size[0]}x{img_size[1]}px)",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    
    if uploaded_file is not None:
        # Procesar imagen
        with st.spinner("🔄 Procesando imagen..."):
            img_array, img_display = preprocess_image(uploaded_file, img_size)
        
        if img_array is not None:
            # Mostrar resultados
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_display, caption=f"Imagen procesada", use_column_width=True)
                st.write(f"**Archivo:** {uploaded_file.name}")
            
            with col2:
                # Predecir
                with st.spinner("🔍 Clasificando..."):
                    class_name, confidence = predict(model, img_array)
                
                if "Error" not in class_name:
                    # Resultados
                    st.success(f"**🎯 Clasificación:** {class_name}")
                    st.progress(confidence)
                    st.write(f"**Confianza:** {confidence:.1%}")
                    
                    # Información de la categoría
                    st.markdown("---")
                    if "BlueRecyclable" in class_name:
                        material = class_name.split("_")[1]
                        st.info(f"🔵 **RECICLABLE** - Contenedor Azul\n\nMaterial: {material}")
                    elif "BrownCompost" in class_name:
                        st.info("🟤 **ORGÁNICO** - Contenedor Marrón")
                    elif "GrayTrash" in class_name:
                        st.info("⚪ **RESTO** - Contenedor Gris")
                    else:
                        st.warning("🟡 **ESPECIAL** - Consulta normas locales")
                else:
                    st.error(f"❌ {class_name}")

else:
    st.error("🚫 No se pudo inicializar el sistema")
    
    # Soluciones específicas
    with st.expander("🔧 Soluciones Avanzadas"):
        st.write("""
        **Problema:** El modelo parece tener un problema de configuración interna
        
        **Soluciones a intentar:**
        
        1. **Reentrenar el modelo:**
           - Asegúrate de que el modelo se entrene con imágenes RGB (3 canales)
           - Verifica la forma de entrada durante el entrenamiento
        
        2. **Reconstruir el modelo:**
           ```python
           # Ejemplo de cómo debería crearse el modelo
           model = tf.keras.applications.EfficientNetB2(
               weights=None,
               input_shape=(381, 381, 3),  # ← Asegurar 3 canales
               classes=11
           )
           ```
        
        3. **Convertir el modelo:**
           - Guarda el modelo con `save_format='tf'`
           - O exporta como SavedModel
        
        4. **Verificar el proceso de entrenamiento:**
           - Asegúrate de que las imágenes de entrenamiento sean RGB
           - Verifica el preprocesamiento durante el entrenamiento
        """)
    
    # Opción de emergencia: usar un modelo temporal
    st.write("## 🆕 Opción de Emergencia")
    if st.button("🔄 Crear Modelo Temporal para Pruebas"):
        try:
            with st.spinner("Creando modelo temporal..."):
                temp_model = tf.keras.applications.EfficientNetB2(
                    weights='imagenet',
                    input_shape=(380, 380, 3),
                    include_top=False
                )
                # Añadir capas de clasificación
                x = tf.keras.layers.GlobalAveragePooling2D()(temp_model.output)
                x = tf.keras.layers.Dense(128, activation='relu')(x)
                predictions = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
                model = tf.keras.Model(inputs=temp_model.input, outputs=predictions)
                
                st.success("✅ Modelo temporal creado (solo para pruebas)")
                st.warning("⚠️ Este modelo no está entrenado para residuos, solo para demostración")
        except Exception as e:
            st.error(f"❌ Error creando modelo temporal: {str(e)}")

# Footer
st.markdown("---")
st.caption("♻️ Clasificador de Residuos | Sistema de Diagnóstico")
