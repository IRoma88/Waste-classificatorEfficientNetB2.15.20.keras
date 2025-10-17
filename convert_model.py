# convert_model.py - EJECUTAR LOCALMENTE
import tensorflow as tf
import os

print("🔄 Convirtiendo modelo para compatibilidad...")

try:
    # Cargar tu modelo original
    model = tf.keras.models.load_model("models/EfficientNetB2.15.20.keras")
    print("✅ Modelo original cargado")
    
    # Guardar en formato SavedModel (más compatible)
    model.save("converted_model", save_format='tf')
    print("✅ Modelo convertido a SavedModel")
    
    # También guardar en formato Keras con compatibilidad
    model.save("converted_model.h5")
    print("✅ Modelo guardado como .h5")
    
    # Verificar
    test_model = tf.keras.models.load_model("converted_model")
    print(f"✅ Modelo convertido verificado: {test_model.input_shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
