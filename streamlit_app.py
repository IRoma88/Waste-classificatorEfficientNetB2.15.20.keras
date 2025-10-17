# streamlit_app.py - VERSIÓN CORREGIDA
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

# --- CARGA OPTIMIZADA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        # Verificar que el archivo existe
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Archivo no encontrado: {MODEL_PATH}")
            st.info("💡 Asegúrate de que el archivo del modelo esté en la carpeta 'models/'")
            return None
        
        # Cargar con configuración específica
        with st.spinner("🔄 Cargando modelo (puede tomar unos segundos)..."):
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )
            
            # Compilar manualmente
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Verificar la forma de entrada esperada
        input_shape = model.input_shape
        st.success(f"✅ Modelo cargado exitosamente")
        st.info(f"📊 Forma de entrada esperada: {input_shape}")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        return None

# Cargar modelo
model = load_model()

# --- FUNCIONES DE PREPROCESAMIENTO CORREGIDAS ---
def ensure_rgb(image):
    """Garantiza que la imagen tenga 3 canales RGB"""
    try:
        # Convertir PIL Image a numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Manejar diferentes formatos de imagen
        if len(img_array.shape) == 2:  # Escala de grises (H, W)
            img_rgb = np.stack([img_array] * 3, axis=-1)
            st.info("🔄 Imagen convertida de escala de grises a RGB")
            
        elif img_array.shape[2] == 1:  # Un solo canal (H, W, 1)
            img_rgb = np.concatenate([img_array] * 3, axis=-1)
            st.info("🔄 Imagen convertida de 1 canal a RGB")
            
        elif img_array.shape[2] == 4:  # RGBA (H, W, 4)
            img_rgb = img_array[:, :, :3]  # Quitar canal alpha
            st.info("🔄 Imagen convertida de RGBA a RGB")
            
        elif img_array.shape[2] == 3:  # Ya es RGB
            img_rgb = img_array
            
        else:
            st.warning(f"⚠️ Formato inesperado: {img_array.shape}")
            # Intentar conversión por defecto
            img_rgb = img_array[:, :, :3] if img_array.shape[2] > 3 else img_array
            
        return img_rgb
        
    except Exception as e:
        st.error(f"❌ Error en conversión RGB: {e}")
        return None

def preprocess_image(uploaded_file):
    """Preprocesa la imagen asegurando formato correcto para EfficientNetB2"""
    try:
        # Abrir imagen
        img = Image.open(uploaded_file)
        
        # Convertir a RGB usando PIL (primer intento)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img_resized = img.resize(IMG_SIZE)
        
        # Convertir a array
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Garantizar 3 canales RGB
        img_array = ensure_rgb(img_array)
        if img_array is None:
            return None, None
        
        # Normalizar a [0,1]
        img_array = img_array / 255.0
        
        # Verificar forma final
        expected_shape = (*IMG_SIZE, 3)
        if img_array.shape != expected_shape:
            st.warning(f"⚠️ Forma final: {img_array.shape}. Esperado: {expected_shape}")
            # Forzar redimensionamiento si es necesario
            if img_array.shape[:2] != IMG_SIZE:
                img_array = tf.image.resize(img_array, IMG_SIZE).numpy()
        
        # Añadir dimensión del batch
        img_array = np.expand_dims(img_array, axis=0)
        
        st.success(f"✅ Imagen procesada: {img_array.shape}")
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
        st.write(f"**Modelo:** EfficientNetB2")
        st.write(f"**Forma de entrada:** {model.input_shape}")
        st.write(f"**Clases:** {len(CLASS_NAMES)} categorías")
        st.write(f"**Resolución:** {IMG_SIZE[0]}x{IMG_SIZE[1]} píxeles")
        st.write(f"**Canales:** RGB (3 canales)")
    
    # Uploader de imagen
    uploaded_file = st.file_uploader(
        "Sube una imagen de residuo para clasificar", 
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Formatos soportados: JPG, JPEG, PNG, WEBP, BMP"
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
                st.write(f"**Forma procesada:** {img_array.shape}")
            
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
                        material = pred_class.split("_")[1]
                        st.write(f"Material: {material} - Deposita en contenedor azul")
                    elif "BrownCompost" in pred_class:
                        st.info("🟤 **CONTENEDOR MARRÓN - Orgánico**")
                        st.write("Restos de comida, frutas, verduras - Deposita en contenedor marrón")
                    elif "GrayTrash" in pred_class:
                        st.info("⚪ **CONTENEDOR GRIS - Resto**")
                        st.write("Materiales no reciclables - Deposita en contenedor gris")
                    elif "SPECIAL" in pred_class:
                        st.warning("🟡 **CATEGORÍA ESPECIAL**")
                        st.write("Consulta las normas específicas de tu municipio para este material")
                    
                else:
                    st.error(f"❌ {pred_class}")

else:
    st.error("🚫 No se pudo inicializar el sistema de clasificación")
    
    # Información de solución de problemas
    with st.expander("🔧 Solución de problemas - DETALLADO"):
        st.write("""
        **Problema específico detectado:**
        - El modelo espera imágenes RGB (3 canales) pero recibe imágenes con 1 canal
        
        **Causas posibles:**
        1. Imágenes en escala de grises o con formato incorrecto
        2. Problema en el preprocesamiento de imágenes
        3. Incompatibilidad de versiones de TensorFlow
        
        **Soluciones aplicadas:**
        ✅ Conversión forzada a RGB
        ✅ Verificación de canales de imagen
        ✅ Manejo de diferentes formatos (RGBA, escala de grises)
        ✅ Validación de forma de entrada
        
        **Si persiste el error:**
        1. Verifica que el archivo del modelo esté completo
        2. Prueba con diferentes imágenes
        3. Verifica los logs de Streamlit Cloud para más detalles
        """)

# Footer
st.markdown("---")
st.caption("♻️ Clasificador de Residuos - EfficientNetB2 | Desarrollado con TensorFlow y Streamlit")
