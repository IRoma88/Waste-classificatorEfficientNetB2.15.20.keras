import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Clasificador de Residuos", page_icon="♻️", layout="centered")
st.title("♻️ Clasificador de Residuos")

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

@st.cache_resource
def load_model():
    try:
        # USA EL MODELO REPARADO que SÍ está en GitHub sin LFS
        model = tf.keras.models.load_model("models/EfficientNetB2_repaired.keras")
        st.success("✅ Modelo cargado exitosamente!")
        return model
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((381, 381))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Cargar modelo
model = load_model()

if model:
    st.success("¡Sistema listo para clasificar!")
    
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)
        
        with st.spinner("Clasificando..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            class_name = CLASS_NAMES[class_idx]
        
        st.success(f"**Resultado:** {class_name}")
        st.progress(confidence)
        st.write(f"**Confianza:** {confidence:.1%}")

else:
    st.error("No se pudo cargar el modelo")

st.markdown("---")
st.caption("Clasificador de Residuos")
