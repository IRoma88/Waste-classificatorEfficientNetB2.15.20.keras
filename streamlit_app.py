# streamlit_app.py - VERSIÓN OPTIMIZADA PARA STREAMLIT
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuración
st.set_page_config(page_title="Clasificador de Residuos", page_icon="♻️", layout="centered")
st.title("♻️ Clasificador de Residuos")

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/EfficientNetB2.15.20.keras"
IMG_SIZE = (380, 380)

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

# --- CARGA OPTIMIZADA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        # Verificar que el archivo existe
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Archivo no encontrado: {MODEL_PATH}")
            return None
        
        # Cargar con configuración específica para compatibilidad
        with st.spinner("🔄 Cargando modelo (puede tomar unos segundos)..."):
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False  # No compilar inicialmente
            )
            
            # Compilar manualmente si es necesario
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        st.success("✅ Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        st.info("💡 Esto puede ser un problema de compatibilidad de versiones de TensorFlow")
        return None

# Cargar modelo
model = load_model()

# --- FUNCIONES DE PREPROCESAMIENTO ---
def preprocess_image(uploaded_file):
    """Preprocesa la imagen asegurando formato correcto"""
    try:
        # Abrir imagen
        img = Image.open(uploaded_file)
        
        # Forzar conversión a RGB (3 canales)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            st.info("🔄 Imagen convertida a RGB")
        
        # Redimensionar
        img_resized = img.resize(IMG_SIZE)
        
        # Convertir a array y normalizar
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Verificar forma
        if img_array.shape != (*IMG_SIZE, 3):
            st.warning(f"⚠️ Forma inesperada: {img_array.shape}")
        
        # Añadir dimensión del batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {e}")
        return None, None

def predict(img_array):
    """Realiza predicción con manejo de errores"""
    try:
        # Verificar forma de entrada
        expected_shape = (1, 380, 380, 3)
        if img_array.shape != expected_shape:
            st.error(f"❌ Forma incorrecta: {img_array.shape}. Esperado: {expected_shape}")
            return "Error: Formato de imagen incorrecto", 0.0
        
        # Realizar predicción
        preds = model.predict(img_array, verbose=0)
        
        # Obtener resultados
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        
        return CLASS_NAMES[class_id], confidence
        
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL ---
if model is not None:
    st.success("✅ ¡Sistema listo para clasificar residuos!")
    
    # Información del sistema
    with st.expander("📊 Información del Sistema"):
        st.write(f"**Modelo:** EfficientNetB2 (54.4 MB)")
        st.write(f"**Clases:** {len(CLASS_NAMES)} categorías")
        st.write(f"**Resolución:** {IMG_SIZE[0]}x{IMG_SIZE[1]} píxeles")
        st.write(f"**Formato de entrada:** RGB (3 canales)")
    
    # Uploader de imagen
    uploaded_file = st.file_uploader(
        "Sube una imagen de residuo para clasificar", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos soportados: JPG, JPEG, PNG, WEBP"
    )
    
    if uploaded_file is not None:
        # Procesar imagen
        img_array, img_display = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Mostrar en dos columnas
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_display, caption="Imagen subida", use_column_width=True)
                st.write(f"**Archivo:** {uploaded_file.name}")
                st.write(f"**Tamaño procesado:** {img_array.shape}")
            
            with col2:
                # Realizar predicción
                with st.spinner("🔍 Analizando imagen..."):
                    pred_class, confidence = predict(img_array)
                
                if "Error" not in pred_class:
                    # Mostrar resultados
                    st.success(f"**🎯 CLASIFICACIÓN:** {pred_class}")
                    
                    # Barra de confianza
                    st.progress(confidence)
                    st.write(f"**📊 CONFIANZA:** {confidence*100:.2f}%")
                    
                    # Información de la categoría
                    st.markdown("---")
                    if "BlueRecyclable" in pred_class:
                        st.info("🔵 **CONTENEDOR AZUL - Reciclable**")
                        st.write("Materiales reciclables: papel, cartón, vidrio, metales, plásticos")
                    elif "BrownCompost" in pred_class:
                        st.info("🟤 **CONTENEDOR MARRÓN - Orgánico**")
                        st.write("Restos de comida, frutas, verduras, materiales compostables")
                    elif "GrayTrash" in pred_class:
                        st.info("⚪ **CONTENEDOR GRIS - Resto**")
                        st.write("Materiales no reciclables ni compostables")
                    elif "SPECIAL" in pred_class:
                        st.warning("🟡 **CATEGORÍA ESPECIAL**")
                        st.write("Consulta las normas específicas de tu municipio")
                    
                    # Interpretación de confianza
                    st.markdown("---")
                    if confidence > 0.8:
                        st.success("🟢 **ALTA CONFIANZA** - Clasificación muy fiable")
                    elif confidence > 0.6:
                        st.info("🟡 **CONFIANZA MEDIA** - Clasificación probablemente correcta")
                    else:
                        st.warning("🔴 **BAJA CONFIANZA** - Considera verificar manualmente")
                        
                else:
                    st.error(f"❌ {pred_class}")

else:
    st.error("🚫 No se pudo inicializar el sistema de clasificación")
    
    # Información de solución de problemas
    with st.expander("🔧 Solución de problemas"):
        st.write("""
        **Problemas comunes:**
        - Incompatibilidad de versiones de TensorFlow
        - Archivo de modelo corrupto
        - Memoria insuficiente en Streamlit Cloud
        
        **Soluciones:**
        1. Verifica que el modelo esté en la carpeta `models/`
        2. Asegúrate de que sea un archivo .keras válido
        3. Recarga la aplicación
        """)

# Footer
st.markdown("---")
st.caption("♻️ Clasificador de Residuos - EfficientNetB2 | Desarrollado con TensorFlow y Streamlit")
