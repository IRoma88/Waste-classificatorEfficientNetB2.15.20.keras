# repair_model_standalone.py
import tensorflow as tf
import numpy as np
import os

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

def repair_model():
    print("🔧 Iniciando reparación del modelo...")
    
    try:
        # Verificar que existe el modelo original
        if not os.path.exists("models/EfficientNetB2.15.20.keras"):
            print("❌ No se encuentra el modelo original")
            return False
        
        # Cargar modelo original
        print("📥 Cargando modelo original...")
        original_model = tf.keras.models.load_model("models/EfficientNetB2.15.20.keras", compile=False)
        print("✅ Modelo original cargado")
        
        # Crear nuevo modelo con arquitectura correcta
        print("🏗️ Creando nueva arquitectura...")
        new_model = tf.keras.applications.EfficientNetB2(
            weights=None,
            input_shape=(381, 381, 3),
            classes=len(CLASS_NAMES)
        )
        
        # Intentar transferir pesos
        print("🔄 Transferiendo pesos...")
        transferred_count = 0
        for new_layer in new_model.layers:
            for orig_layer in original_model.layers:
                if new_layer.name == orig_layer.name:
                    try:
                        new_layer.set_weights(orig_layer.get_weights())
                        transferred_count += 1
                        print(f"✅ Transferidos pesos de: {new_layer.name}")
                        break
                    except Exception as e:
                        print(f"⚠️ Error transfiriendo {new_layer.name}: {e}")
                        continue
        
        print(f"📊 Total de capas transferidas: {transferred_count}/{len(new_model.layers)}")
        
        # Compilar
        new_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Guardar modelo reparado
        print("💾 Guardando modelo reparado...")
        new_model.save("models/EfficientNetB2_repaired.keras")
        print("✅ Modelo reparado guardado exitosamente")
        
        # Verificar
        test_model = tf.keras.models.load_model("models/EfficientNetB2_repaired.keras")
        print(f"📊 Forma de entrada del modelo reparado: {test_model.input_shape}")
        print(f"📊 Forma de salida del modelo reparado: {test_model.output_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la reparación: {e}")
        return False

if __name__ == "__main__":
    repair_model()
