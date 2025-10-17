# streamlit_real_model.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Clasificador de Residuos", page_icon="♻️", layout="centered")
st.title("♻️ Clasificador de Residuos - MODELO REAL")

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

@st.cache_resource
def load_real_model():
    """Cargar tu modelo entrenado real"""
    try:
        st.info("🔄 Cargando modelo entrenado...")
        
        # Intentar diferentes formatos
        model_paths = [
            "converted_model",           # SavedModel format
            "converted_model.h5",        # H5 format  
            "models/EfficientNetB2.15.20.keras"  # Original
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path)
                    st.success(f"✅ Modelo cargado: {path}")
                    st.info(f"📊 Input shape: {model.input_shape}")
                    return model
                except Exception as e:
                    st.warning(f"⚠️ No se pudo cargar {path}: {e}")
                    continue
        
        st.error("❌ No se pudo cargar ningún modelo")
        return None
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {e}")
        return None

def preprocess_image(image, target_size):
    """Preprocesamiento para tu modelo específico"""
    # Convertir a RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar al tamaño que espera tu modelo
    image = image.resize(target_size)
    
    # Convertir a array y normalizar
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Añadir dimensión del batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Cargar modelo REAL
model = load_real_model()

if model:
    # Obtener tamaño de entrada del modelo
    if model.input_shape[1] is not None:
        target_size = (model.input_shape[2], model.input_shape[1])  # (width, height)
    else:
        target_size = (381, 381)  # Tamaño por defecto de EfficientNetB2
    
    st.success(f"✅ ¡Modelo real cargado! Tamaño: {target_size[0]}x{target_size[1]}")
    
    # Subir imagen
    uploaded_file = st.file_uploader(
        f"Sube una imagen (se redimensionará a {target_size[0]}x{target_size[1]})",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen original", use_column_width=True)
        
        # Procesar y predecir
        with st.spinner("🔍 Clasificando con modelo entrenado..."):
            try:
                # Preprocesar
                img_array = preprocess_image(image, target_size)
                
                # Predecir con tu modelo REAL
                predictions = model.predict(img_array, verbose=0)[0]
                
                # Obtener top 3 resultados
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                top_3_classes = [CLASS_NAMES[i] for i in top_3_idx]
                top_3_confidences = [predictions[i] for i in top_3_idx]
                
                # Resultado principal
                main_class = top_3_classes[0]
                main_confidence = top_3_confidences[0]
                
                # Mostrar resultados
                st.success(f"**🎯 CLASIFICACIÓN:** {main_class}")
                st.progress(main_confidence)
                st.write(f"**📊 CONFIANZA:** {main_confidence:.1%}")
                
                # Top 3 predicciones
                st.write("**🏆 TOP 3 PREDICCIONES:**")
                for i, (cls, conf) in enumerate(zip(top_3_classes, top_3_confidences)):
                    st.write(f"{i+1}. {cls} ({conf:.1%})")
                
                # Información detallada
                st.markdown("---")
                if "BlueRecyclable" in main_class:
                    material = main_class.split("_")[1]
                    st.info(f"🔵 **RECICLABLE** - {material}")
                elif "BrownCompost" in main_class:
                    st.info("🟤 **ORGÁNICO** - Contenedor marrón")
                elif "GrayTrash" in main_class:
                    st.info("⚪ **RESTO** - Contenedor gris")
                else:
                    st.warning("🟡 **ESPECIAL** - Consultar normas")
                    
            except Exception as e:
                st.error(f"❌ Error en predicción: {e}")

else:
    st.error("No se pudo cargar el modelo entrenado")
    st.info("💡 Ejecuta 'convert_model.py' localmente y sube los modelos convertidos")

st.markdown("---")
st.caption("♻️ Clasificador con Modelo Entrenado Real")
