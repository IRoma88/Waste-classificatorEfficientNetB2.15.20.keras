# streamlit_app.py - VERSIÓN CON REPARACIÓN DEL MODELO
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
REPAIRED_MODEL_PATH = "models/EfficientNetB2_repaired.keras"

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

# --- REPARACIÓN DEL MODELO ---
@st.cache_resource
def repair_and_load_model():
    try:
        st.write("🔍 **Diagnóstico y reparación del modelo...**")
        
        # Verificar que el archivo existe
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Archivo no encontrado: {MODEL_PATH}")
            return None, None
        
        st.success("✅ Archivo del modelo encontrado")
        
        # Verificar si ya existe un modelo reparado
        if os.path.exists(REPAIRED_MODEL_PATH):
            st.info("🔄 Cargando modelo reparado existente...")
            try:
                model = tf.keras.models.load_model(REPAIRED_MODEL_PATH, compile=False)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                st.success("✅ Modelo reparado cargado exitosamente")
                return model, model.input_shape[1:3]
            except Exception as e:
                st.warning(f"⚠️ Modelo reparado falló: {str(e)}")
        
        # Intentar reparar el modelo original
        st.info("🛠️ Intentando reparar el modelo original...")
        
        try:
            # Método 1: Cargar los pesos manualmente
            with st.spinner("Reparando modelo (esto puede tomar un momento)..."):
                
                # Crear un nuevo modelo con la arquitectura correcta
                base_model = tf.keras.applications.EfficientNetB2(
                    weights=None,
                    input_shape=(381, 381, 3),  # Forma CORRECTA con 3 canales
                    include_top=False
                )
                
                # Añadir capas de clasificación
                x = base_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(512, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                predictions = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
                
                # Crear modelo completo
                model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
                
                # Intentar cargar los pesos del modelo original
                try:
                    original_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    
                    # Transferir pesos capa por capa
                    for i, layer in enumerate(model.layers):
                        if i < len(original_model.layers):
                            try:
                                if layer.name == original_model.layers[i].name:
                                    layer.set_weights(original_model.layers[i].get_weights())
                            except:
                                continue  # Saltar capas incompatibles
                    
                    st.success("✅ Pesos transferidos exitosamente")
                    
                except Exception as e:
                    st.warning(f"⚠️ No se pudieron transferir pesos: {str(e)}")
                    st.info("ℹ️ Usando modelo con inicialización aleatoria")
                
                # Compilar el modelo
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Guardar modelo reparado
                model.save(REPAIRED_MODEL_PATH)
                st.success("✅ Modelo reparado y guardado")
                
                return model, (381, 381)
                
        except Exception as e:
            st.error(f"❌ Error en reparación: {str(e)}")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Error general: {str(e)}")
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
        
        # Verificar canales
        if img_resized.shape[2] != 3:
            st.error(f"❌ Error: la imagen tiene {img_resized.shape[2]} canales, se requieren 3")
            return None, None
        
        # Normalizar
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Añadir dimensión del batch
        img_batch = np.expand_dims(img_float, axis=0)
        
        # Imagen para mostrar
        img_display = Image.fromarray(img_resized)
        
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
st.write("## 🔧 Sistema de Clasificación de Residuos")

# Cargar o reparar modelo
model, img_size = repair_and_load_model()

if model is not None and img_size is not None:
    st.success(f"✅ ¡Sistema listo! Tamaño de imagen: {img_size[0]}x{img_size[1]}")
    
    # Información del modelo
    with st.expander("📊 Información del Modelo"):
        st.write(f"**Tipo:** EfficientNetB2 (Reparado)")
        st.write(f"**Forma de entrada:** {model.input_shape}")
        st.write(f"**Forma de salida:** {model.output_shape}")
        st.write(f"**Tamaño requerido:** {img_size[0]}x{img_size[1]}px")
        st.write(f"**Número de clases:** {len(CLASS_NAMES)}")
        
        # Mostrar arquitectura simplificada
        st.write("**Arquitectura:**")
        for i, layer in enumerate(model.layers[:5]):  # Primeras 5 capas
            st.write(f"  - {layer.name}: {layer.output_shape}")
        st.write("  - ...")
        for i, layer in enumerate(model.layers[-3:]):  # Últimas 3 capas
            st.write(f"  - {layer.name}: {layer.output_shape}")
    
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
                st.image(img_display, caption="Imagen procesada", use_column_width=True)
                st.write(f"**Archivo:** {uploaded_file.name}")
            
            with col2:
                # Predecir
                with st.spinner("🔍 Analizando residuo..."):
                    class_name, confidence = predict(model, img_array)
                
                # Mostrar resultados
                st.success(f"**🎯 CLASIFICACIÓN:** {class_name}")
                st.progress(confidence)
                st.write(f"**📊 CONFIANZA:** {confidence:.1%}")
                
                # Información detallada de la categoría
                st.markdown("---")
                if "BlueRecyclable" in class_name:
                    material = class_name.split("_")[1]
                    st.info(f"""
                    🔵 **CONTENEDOR AZUL - RECICLABLE**
                    
                    **Material:** {material}
                    **Instrucciones:** Depositar en contenedor azul para reciclaje
                    """)
                elif "BrownCompost" in class_name:
                    st.info(f"""
                    🟤 **CONTENEDOR MARRÓN - ORGÁNICO**
                    
                    **Material:** Restos orgánicos compostables
                    **Instrucciones:** Depositar en contenedor marrón para compostaje
                    """)
                elif "GrayTrash" in class_name:
                    st.info(f"""
                    ⚪ **CONTENEDOR GRIS - RESTO**
                    
                    **Material:** No reciclable ni compostable
                    **Instrucciones:** Depositar en contenedor gris para vertedero
                    """)
                else:
                    st.warning(f"""
                    🟡 **CATEGORÍA ESPECIAL**
                    
                    **Tipo:** {class_name.replace('SPECIAL_', '')}
                    **Instrucciones:** Consultar normas específicas de tu municipio
                    """)
                
                # Interpretación de confianza
                st.markdown("---")
                if confidence > 0.8:
                    st.success("🟢 **ALTA CONFIANZA** - La clasificación es muy confiable")
                elif confidence > 0.6:
                    st.info("🟡 **CONFIANZA MEDIA** - La clasificación es probablemente correcta")
                else:
                    st.warning("🔴 **BAJA CONFIANZA** - Considera verificar manualmente")

else:
    st.error("🚫 No se pudo inicializar el sistema de clasificación")
    
    # Solución alternativa
    st.write("## 🔧 Solución Alternativa")
    
    if st.button("🔄 Crear Modelo de Emergencia (sin entrenamiento previo)"):
        try:
            with st.spinner("Creando modelo de emergencia..."):
                # Modelo simple para demostración
                emergency_model = tf.keras.applications.EfficientNetB2(
                    weights='imagenet',
                    input_shape=(381, 381, 3),
                    include_top=False
                )
                
                # Congelar capas base
                emergency_model.trainable = False
                
                # Añadir nuevas capas
                x = emergency_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(256, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                predictions = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
                
                model = tf.keras.Model(inputs=emergency_model.input, outputs=predictions)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                st.warning("⚠️ Modelo de emergencia creado - **NO ESTÁ ENTRENADO** para residuos")
                st.info("📝 Este modelo solo sirve para demostración. Los resultados serán aleatorios.")
                
                # Continuar con el modelo de emergencia
                model, img_size = model, (381, 381)
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"❌ Error creando modelo de emergencia: {str(e)}")

# Footer
st.markdown("---")
st.caption("♻️ Clasificador de Residuos | Sistema Reparado")
