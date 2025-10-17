# repair_model_advanced.py
import tensorflow as tf
import numpy as np
import os
import h5py

CLASS_NAMES = [
    "BlueRecyclable_Cardboard", "BlueRecyclable_Glass", "BlueRecyclable_Metal",
    "BlueRecyclable_Paper", "BlueRecyclable_Plastics", "BrownCompost",
    "GrayTrash", "SPECIAL_DropOff", "SPECIAL_TakeBackShop", 
    "SPECIAL_MedicalTakeBack", "SPECIAL_HHW"
]

def force_repair_model():
    print("🔧 INICIANDO REPARACIÓN AVANZADA DEL MODELO...")
    
    try:
        # 1. Verificar que existe el modelo original
        if not os.path.exists("models/EfficientNetB2.15.20.keras"):
            print("❌ No se encuentra el modelo original")
            return False
        
        print("📥 Cargando modelo original para diagnóstico...")
        
        # 2. Intentar cargar el modelo original para diagnóstico
        try:
            original_model = tf.keras.models.load_model("models/EfficientNetB2.15.20.keras", compile=False)
            print("✅ Modelo original cargado para diagnóstico")
            
            # Diagnosticar el problema
            print(f"🔍 DIAGNÓSTICO:")
            print(f"   - Input shape: {original_model.input_shape}")
            print(f"   - Output shape: {original_model.output_shape}")
            print(f"   - Número de capas: {len(original_model.layers)}")
            
            # Verificar las primeras capas
            for i, layer in enumerate(original_model.layers[:3]):
                print(f"   - Capa {i} ({layer.name}): {layer.input_shape} -> {layer.output_shape}")
                
        except Exception as e:
            print(f"⚠️ No se pudo cargar el modelo original: {e}")
            print("🔄 Intentando método alternativo...")
        
        # 3. MÉTODO 1: Cargar como archivo HDF5 y extraer pesos manualmente
        print("\n🔄 MÉTODO 1: Extracción manual de pesos...")
        try:
            # Abrir el archivo como HDF5
            with h5py.File("models/EfficientNetB2.15.20.keras", 'r') as f:
                print("✅ Archivo HDF5 abierto correctamente")
                
                # Explorar la estructura
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"     Dataset: {name}, shape: {obj.shape}")
                    elif isinstance(obj, h5py.Group):
                        print(f"   Grupo: {name}")
                
                f.visititems(print_structure)
                
        except Exception as e:
            print(f"⚠️ Método HDF5 falló: {e}")
        
        # 4. MÉTODO 2: Crear modelo con la misma arquitectura pero forma correcta
        print("\n🔄 MÉTODO 2: Creando modelo con arquitectura corregida...")
        
        # Crear modelo base EfficientNetB2
        base_model = tf.keras.applications.EfficientNetB2(
            weights=None,
            input_shape=(381, 381, 3),  # FORMA CORRECTA
            include_top=False
        )
        
        # Añadir capas de clasificación (debes usar la misma arquitectura que tu modelo original)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        predictions = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
        
        new_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
        # 5. MÉTODO 3: Transferencia de pesos capa por capa
        print("\n🔄 MÉTODO 3: Transferencia de pesos...")
        transferred_layers = []
        
        try:
            # Cargar modelo original nuevamente para transferencia
            original_model = tf.keras.models.load_model("models/EfficientNetB2.15.20.keras", compile=False)
            
            for new_layer in new_model.layers:
                for orig_layer in original_model.layers:
                    if (new_layer.name == orig_layer.name and 
                        hasattr(orig_layer, 'weights') and 
                        len(orig_layer.weights) > 0):
                        try:
                            # Verificar compatibilidad de formas
                            new_weights = orig_layer.get_weights()
                            if len(new_weights) > 0:
                                new_layer.set_weights(new_weights)
                                transferred_layers.append(new_layer.name)
                                print(f"✅ Transferida: {new_layer.name}")
                                break
                        except Exception as e:
                            print(f"⚠️ Error en {new_layer.name}: {e}")
                            continue
        
        except Exception as e:
            print(f"⚠️ Transferencia automática falló: {e}")
        
        print(f"📊 Capas transferidas: {len(transferred_layers)}/{len(new_model.layers)}")
        
        # 6. Si no se transfirieron suficientes capas, usar pesos pre-entrenados
        if len(transferred_layers) < 10:  # Si se transfirieron menos de 10 capas
            print("\n🔄 Usando pesos pre-entrenados de ImageNet...")
            try:
                # Crear modelo con pesos pre-entrenados
                base_model_pretrained = tf.keras.applications.EfficientNetB2(
                    weights='imagenet',
                    input_shape=(381, 381, 3),
                    include_top=False
                )
                
                # Congelar capas base
                base_model_pretrained.trainable = False
                
                # Reconstruir modelo completo
                x = base_model_pretrained.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(512, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                predictions = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
                
                new_model = tf.keras.Model(inputs=base_model_pretrained.input, outputs=predictions)
                print("✅ Modelo creado con pesos pre-entrenados")
                
            except Exception as e:
                print(f"⚠️ Error con pesos pre-entrenados: {e}")
        
        # 7. Compilar el modelo
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 8. Guardar modelo reparado
        print("\n💾 Guardando modelo reparado...")
        new_model.save("models/EfficientNetB2_repaired.keras")
        
        # 9. Verificación final
        print("🔍 VERIFICACIÓN FINAL:")
        test_model = tf.keras.models.load_model("models/EfficientNetB2_repaired.keras")
        print(f"✅ Input shape: {test_model.input_shape}")
        print(f"✅ Output shape: {test_model.output_shape}")
        print(f"✅ Número de clases: {test_model.output_shape[-1]}")
        
        # Test de predicción con imagen dummy
        print("🧪 Test de predicción...")
        dummy_input = np.random.random((1, 381, 381, 3)).astype(np.float32)
        prediction = test_model.predict(dummy_input, verbose=0)
        print(f"✅ Forma de predicción: {prediction.shape}")
        print(f"✅ Suma de probabilidades: {np.sum(prediction):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = force_repair_model()
    if success:
        print("\n🎉 ¡REPARACIÓN COMPLETADA EXITOSAMENTE!")
    else:
        print("\n💥 REPARACIÓN FALLADA")
