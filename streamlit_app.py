import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración simple
st.set_page_config(page_title="Clasificador de Residuos", page_icon="♻️", layout="centered")
st.title("♻️ Clasificador de Residuos")

# Clases
CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/EfficientNetB2.15.20.keras")
        st.success("✅ Modelo cargado")
        return model
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# Preprocesar imagen
def preprocess_image(image):
    # Convertir a RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar a 381x381 (tamaño que espera tu modelo)
    image = image.resize((381, 381))
    
    # Convertir a array y normalizar
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Añadir dimensión del batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Interfaz principal
model = load_model()

if model:
    st.write("### 📤 Sube una imagen para clasificar")
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)
        
        # Procesar y predecir
        with st.spinner("🔍 Clasificando..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)[0]
            
            # Obtener resultado
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            class_name = CLASS_NAMES[class_idx]
        
        # Mostrar resultados
        st.success(f"**🎯 Resultado:** {class_name}")
        st.progress(confidence)
        st.write(f"**📊 Confianza:** {confidence:.1%}")
        
        # Información simple
        if "BlueRecyclable" in class_name:
            st.info("🔵 **Reciclable** - Contenedor Azul")
        elif "BrownCompost" in class_name:
            st.info("🟤 **Orgánico** - Contenedor Marrón") 
        elif "GrayTrash" in class_name:
            st.info("⚪ **Resto** - Contenedor Gris")
        else:
            st.warning("🟡 **Especial** - Consultar normas")

else:
    st.error("No se pudo cargar el modelo")

st.markdown("---")
st.caption("Clasificador de Residuos - EfficientNetB2")
