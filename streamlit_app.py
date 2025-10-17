# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuración de página / Page configuration
st.set_page_config(
    page_title="Clasificador de Residuos / Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# Título bilingüe / Bilingual title
st.title("♻️ Clasificador de Residuos / Waste Classifier")
st.write("Sube una imagen de residuo para clasificarlo / Upload a waste image to classify it")

# --- CONFIGURACIÓN / CONFIGURATION ---
MODEL_PATH = "https://drive.google.com/file/d/1PcSynIU3Od_82zdHOerJRx3NLyEYbAUH/view?usp=sharing"
IMG_SIZE = (380, 380)

# Clases / Classes (AJUSTA SEGÚN TU ENTRENAMIENTO / ADJUST ACCORDING TO YOUR TRAINING)
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

# --- CARGA DEL MODELO / MODEL LOADING ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modelo no encontrado en / Model not found at: {MODEL_PATH}")
        st.info("💡 Asegúrate de que el archivo del modelo esté en la carpeta 'models/' / Make sure the model file is in the 'models/' folder")
        return None
    
    try:
        with st.spinner("🔄 Cargando modelo... / Loading model..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error cargando modelo / Error loading model: {e}")
        return None

# Cargar modelo al inicio / Load model at startup
model = load_model()

# --- FUNCIONES / FUNCTIONS ---
def preprocess_image(uploaded_file):
    """Preprocesa la imagen para el modelo / Preprocess image for the model"""
    try:
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    except Exception as e:
        st.error(f"❌ Error procesando imagen / Error processing image: {e}")
        return None, None

def predict(img_array):
    """Realiza predicción con el modelo / Make prediction with the model"""
    try:
        preds = model.predict(img_array, verbose=0)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL / MAIN INTERFACE ---
if model is not None:
    st.success("✅ Modelo cargado - ¡Listo para clasificar! / Model loaded - Ready to classify!")
    
    # Información del modelo / Model information
    with st.expander("📊 Información del Modelo / Model Information"):
        st.write(f"**Arquitectura / Architecture:** EfficientNetB2")
        st.write(f"**Épocas de entrenamiento / Training epochs:** 15-20")
        st.write(f"**Tamaño de entrada / Input size:** 380x380 px")
        st.write(f"**Clases / Classes:** {len(CLASS_NAMES)} categorías / categories")
        
        # Mostrar todas las clases / Show all classes
        st.write("**Lista de clases / Class list:**")
        for i, class_name in enumerate(CLASS_NAMES, 1):
            st.write(f"{i}. {class_name}")
    
    # Uploader de imagen / Image uploader
    uploaded_file = st.file_uploader(
        "Sube una imagen / Upload an image", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos soportados / Supported formats: JPG, JPEG, PNG, WEBP"
    )
    
    if uploaded_file is not None:
        # Procesar imagen / Process image
        img_array, img_display = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Mostrar imagen / Display image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_display, caption="Imagen subida / Uploaded image", use_column_width=True)
                
                # Información de la imagen / Image information
                st.write(f"**Nombre archivo / File name:** {uploaded_file.name}")
                st.write(f"**Tipo / Type:** {uploaded_file.type}")
                st.write(f"**Tamaño / Size:** {uploaded_file.size} bytes")
            
            with col2:
                # Realizar predicción / Make prediction
                with st.spinner("🔍 Clasificando... / Classifying..."):
                    pred_class, confidence = predict(img_array)
                
                if "Error" not in pred_class:
                    # Mostrar resultados / Display results
                    st.success(f"**🎯 Predicción / Prediction:** {pred_class}")
                    
                    # Barra de confianza / Confidence bar
                    st.progress(confidence)
                    st.write(f"**📊 Confianza / Confidence:** {confidence*100:.2f}%")
                    
                    # Información de la categoría / Category information
                    st.markdown("---")
                    
                    if "BlueRecyclable" in pred_class:
                        st.info("""
                        🔵 **Contenedor Azul - Reciclable / Blue Container - Recyclable**
                        
                        Materiales como papel, cartón, vidrio, metales y plásticos /
                        Materials like paper, cardboard, glass, metals and plastics
                        """)
                    elif "BrownCompost" in pred_class:
                        st.info("""
                        🟤 **Contenedor Marrón - Orgánico / Brown Container - Organic**
                        
                        Restos de comida, frutas, verduras y materiales compostables /
                        Food scraps, fruits, vegetables and compostable materials
                        """)
                    elif "GrayTrash" in pred_class:
                        st.info("""
                        ⚪ **Contenedor Gris - Resto / Gray Container - General Waste**
                        
                        Materiales no reciclables ni compostables /
                        Non-recyclable and non-compostable materials
                        """)
                    elif "SPECIAL" in pred_class:
                        st.warning("""
                        🟡 **Categoría Especial / Special Category**
                        
                        Consulta las normas específicas de tu municipio para estos residuos /
                        Check your municipality's specific rules for these wastes
                        """)
                    
                    # Interpretación de la confianza / Confidence interpretation
                    st.markdown("---")
                    if confidence > 0.8:
                        st.success("🟢 **Alta confianza / High confidence** - La clasificación es muy fiable / The classification is very reliable")
                    elif confidence > 0.6:
                        st.info("🟡 **Confianza media / Medium confidence** - La clasificación es probablemente correcta / The classification is probably correct")
                    else:
                        st.warning("🔴 **Baja confianza / Low confidence** - Considera verificar manualmente / Consider manual verification")
                        
                else:
                    st.error(f"❌ {pred_class}")

else:
    st.error("🚫 No se pudo cargar el modelo. Revisa la configuración. / Could not load model. Check configuration.")

# Sección de instrucciones / Instructions section
with st.expander("ℹ️ Cómo usar / How to use"):
    st.markdown("""
    ### 📸 Instrucciones / Instructions:
    
    1. **Sube una imagen** / **Upload an image**: Haz clic en 'Browse files' o arrastra una imagen
    2. **Espera el análisis** / **Wait for analysis**: El modelo procesará la imagen automáticamente
    3. **Revisa los resultados** / **Check results**: Verás la categoría y nivel de confianza
    
    ### 💡 Consejos para mejores resultados / Tips for better results:
    - Usa imágenes con buena iluminación / Use well-lit images
    - Enfoca claramente el objeto / Focus clearly on the object
    - Toma la foto sobre fondo neutro / Take photo on neutral background
    - Evita imágenes borrosas o oscuras / Avoid blurry or dark images
    
    ### 🗑️ Sobre las categorías / About categories:
    - **🔵 Azul/Blue**: Reciclables / Recyclables
    - **🟤 Marrón/Brown**: Orgánico / Organic
    - **⚪ Gris/Gray**: Resto / General waste
    - **🟡 Especial/Special**: Residuos específicos / Specific wastes
    """)

# Footer
st.markdown("---")
st.caption("♻️ Clasificador de Residuos con EfficientNetB2 | Waste Classifier with EfficientNetB2")
st.caption("Desarrollado con TensorFlow y Streamlit / Developed with TensorFlow and Streamlit")
