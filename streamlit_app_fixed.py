# streamlit_app_fixed.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

# Configuración
st.set_page_config(page_title="Clasificador de Residuos", page_icon="♻️", layout="centered")
st.title("♻️ Clasificador de Residuos Inteligente")

# --- CONFIGURACIÓN ---
REPAIRED_MODEL_PATH = "models/EfficientNetB2_repaired.keras"

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

# --- CARGA DEL MODELO REPARADO ---
@st.cache_resource
def load_repaired_model():
    try:
        if not os.path.exists(REPAIRED_MODEL_PATH):
            st.error(f"❌ No se encuentra el modelo reparado: {REPAIRED_MODEL_PATH}")
            st.info("💡 Ejecuta primero: python repair_model_advanced.py")
            return None, None
        
        with st.spinner("🔄 Cargando modelo reparado..."):
            model = tf.keras.models.load_model(REPAIRED_MODEL_PATH, compile=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Obtener tamaño de imagen del modelo
        img_size = model.input_shape[1:3]
        
        st.success("✅ Modelo reparado cargado exitosamente")
        st.info(f"📊 Tamaño de entrada: {img_size[0]}x{img_size[1]}")
        
        return model, img_size
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo reparado: {str(e)}")
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
        
        # Manejar diferentes formatos
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = img[:, :, :3]
        
        # Redimensionar
        img_resized = cv2.resize(img_rgb, (img_size[1], img_size[0]))
        
        # Verificar canales
        if img_resized.shape[2] != 3:
            st.error(f"❌ Error: la imagen tiene {img_resized.shape[2]} canales")
            return None, None
        
        # Normalizar para EfficientNet
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Añadir dimensión del batch
        img_batch = np.expand_dims(img_float, axis=0)
        
        # Imagen para mostrar
        img_display = Image.fromarray(img_resized)
        
        return img_batch, img_display
        
    except Exception as e:
        st.error(f"❌ Error en preprocesamiento: {str(e)}")
        return None, None

def predict_with_confidence(model, img_array):
    """Realizar predicción con análisis de confianza"""
    try:
        # Predicción
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Obtener top 3 predicciones
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_classes = [CLASS_NAMES[i] for i in top_3_indices]
        top_3_confidences = [predictions[i] for i in top_3_indices]
        
        # Clase principal
        main_class = top_3_classes[0]
        main_confidence = top_3_confidences[0]
        
        return main_class, main_confidence, top_3_classes, top_3_confidences
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0, [], []

# --- INTERFAZ PRINCIPAL ---
st.write("## 🤖 Sistema de Clasificación Avanzado")

# Cargar modelo
model, img_size = load_repaired_model()

if model is not None and img_size is not None:
    st.success(f"✅ ¡Sistema listo! Imágenes: {img_size[0]}x{img_size[1]}px")
    
    # Información del sistema
    with st.expander("📊 Información del Sistema"):
        st.write(f"**Modelo:** EfficientNetB2 Reparado")
        st.write(f"**Resolución:** {img_size[0]}x{img_size[1]} píxeles")
        st.write(f"**Clases disponibles:** {len(CLASS_NAMES)}")
        st.write(f"**Arquitectura:** CNN con transfer learning")
    
    # Subir imagen
    st.write("## 📤 Clasificar Imagen")
    uploaded_file = st.file_uploader(
        f"Sube una imagen de residuo",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="La imagen se redimensionará automáticamente a 381x381 píxeles"
    )
    
    if uploaded_file is not None:
        # Procesar imagen
        with st.spinner("🔄 Procesando imagen..."):
            img_array, img_display = preprocess_image(uploaded_file, img_size)
        
        if img_array is not None:
            # Mostrar en columnas
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_display, caption="Imagen procesada", use_column_width=True)
                st.write(f"**Archivo:** {uploaded_file.name}")
                st.write(f"**Formato:** {img_size[0]}x{img_size[1]} RGB")
            
            with col2:
                # Predecir
                with st.spinner("🔍 Analizando composición del residuo..."):
                    main_class, main_confidence, top_3_classes, top_3_confidences = predict_with_confidence(model, img_array)
                
                if "Error" not in main_class:
                    # Resultado principal
                    st.success(f"**🎯 CLASIFICACIÓN PRINCIPAL:** {main_class}")
                    
                    # Barra de confianza
                    st.progress(main_confidence)
                    st.write(f"**📊 CONFIANZA:** {main_confidence:.1%}")
                    
                    # Top 3 predicciones
                    st.write("**🏆 TOP 3 CLASIFICACIONES:**")
                    for i, (cls, conf) in enumerate(zip(top_3_classes, top_3_confidences)):
                        st.write(f"{i+1}. {cls} ({conf:.1%})")
                    
                    # Información de la categoría
                    st.markdown("---")
                    if "BlueRecyclable" in main_class:
                        material = main_class.split("_")[1]
                        st.info(f"""
                        🔵 **CONTENEDOR AZUL - RECICLABLE**
                        
                        **Material:** {material}
                        **Instrucciones:** Depositar en contenedor azul para reciclaje
                        **Beneficio:** Reduce el consumo de recursos naturales
                        """)
                    elif "BrownCompost" in main_class:
                        st.info(f"""
                        🟤 **CONTENEDOR MARRÓN - ORGÁNICO**
                        
                        **Material:** Restos compostables
                        **Instrucciones:** Depositar en contenedor marrón
                        **Beneficio:** Se convierte en abono natural
                        """)
                    elif "GrayTrash" in main_class:
                        st.info(f"""
                        ⚪ **CONTENEDOR GRIS - RESTO**
                        
                        **Material:** No reciclable
                        **Instrucciones:** Depositar en contenedor gris
                        **Impacto:** Va a vertedero controlado
                        """)
                    else:
                        st.warning(f"""
                        🟡 **CATEGORÍA ESPECIAL**
                        
                        **Tipo:** {main_class.replace('SPECIAL_', '')}
                        **Instrucciones:** Consultar punto limpio municipal
                        **Nota:** Requiere gestión especializada
                        """)
                    
                    # Interpretación de confianza
                    st.markdown("---")
                    if main_confidence > 0.7:
                        st.success("🟢 **ALTA CONFIANZA** - Clasificación muy confiable")
                    elif main_confidence > 0.4:
                        st.info("🟡 **CONFIANZA MEDIA** - Clasificación probable")
                    else:
                        st.warning("🔴 **BAJA CONFIANZA** - Verificar manualmente")
                        
                else:
                    st.error(f"❌ {main_class}")

else:
    st.error("🚫 Sistema no disponible")
    
    # Instrucciones de reparación
    st.write("## 🔧 Instrucciones de Reparación")
    st.code("""
# Ejecuta en tu terminal:
python repair_model_advanced.py

# Esto creará un modelo reparado en:
# models/EfficientNetB2_repaired.keras
    """)

# Footer
st.markdown("---")
st.caption("♻️ Clasificador Inteligente de Residuos | v2.0 Reparado")
