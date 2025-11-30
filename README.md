<h1 align="center"> ♻️ Clasificador de Residuos – Waste Classifier </h1>

Este proyecto es un clasificador de residuos basado en **EfficientNetB2**, entrenado durante **15–20 épocas** con un conjunto variado de residuos.  
El objetivo es identificar distintos tipos de materiales (plástico, vidrio, papel, metal, orgánico, etc.) tanto en **imágenes** como en **vídeos**, a través de dos aplicaciones interactivas en Streamlit.

---

This project is a **waste classification system** powered by **EfficientNetB2**, trained for **15–20 epochs** with a diverse dataset of recyclable and non-recyclable materials.  
The goal is to identify different waste types (plastic, glass, paper, metal, organic, etc.) in both **images** and **videos**, using two interactive Streamlit applications.

---

## 🔹 Clasificación de imágenes / Image Classification  
https://waste-classificatorefficientnetb21520keras-6xsvvsvdwcqdarvmarw.streamlit.app/

## 🔹 Clasificación de vídeos / Video Classification (frame-by-frame with motion detection)  
https://waste-classificatorefficientnetb21520keras-d556wx5b4k7adjtt2hn.streamlit.app/

---

# 🧩 ¿Qué hace este proyecto? / What does this project do?

### ES
- Carga un modelo **EfficientNetB2** guardado en formato *SavedModel*.  
- Permite subir imágenes y obtener la predicción del tipo de residuo.  
- Permite subir vídeos cortos (**≤ 15 s**) y analizar cuadro por cuadro:  
  - detección de movimiento,  
  - cajas delimitadoras,  
  - clase predicha,  
  - porcentaje de confianza,  
  - panel lateral con conteo total por tipo de residuo.  
- Genera archivos **CSV** con las predicciones por cuadro y un resumen final.  
- Está diseñado para funcionar en **Streamlit Cloud** con recursos limitados.

**Limitaciones técnicas reales en Streamlit Cloud (entorno gratuito):**  
- Duración máxima recomendada del vídeo: **≤ 15 segundos**  
- Tamaño máximo recomendado del archivo: **50–80 MB**  
- CPU limitada (**1 core**)  
- Memoria disponible: **~1 GB de RAM**  
- Sin GPU disponible  

---

### EN
- Loads an **EfficientNetB2** model stored as a *SavedModel*.  
- Allows users to upload images and receive a predicted waste category.  
- Allows users to upload short videos (**≤ 15 s**) and performs frame-by-frame analysis including:  
  - motion detection,  
  - bounding boxes,  
  - predicted class,  
  - confidence score,  
  - a side panel summarizing counts per class.  
- Generates **CSV** files with per-frame predictions and a summary.  
- Designed to run on **Streamlit Cloud** with limited hardware resources.

**Real technical limitations of free Streamlit Cloud:**  
- Maximum recommended video duration: **≤ 15 seconds**  
- Recommended max file size: **50–80 MB**  
- Limited CPU (**1 core**)  
- Available memory: **~1 GB RAM**  
- No GPU  

---

# ✨ ¿Qué he hecho yo exactamente? / What did I actually build?

### ES
He construido el proyecto paso a paso: primero el modelo de clasificación de imágenes y, una vez que funcionaba bien, desarrollé un sistema completo de análisis cuadro por cuadro para vídeo. Añadí detección de movimiento, cajas delimitadoras, conteo de residuos y un panel visual informativo. Finalmente, integré todo en dos aplicaciones de Streamlit para que cualquiera pueda probarlo subiendo sus propias imágenes o vídeos.

### EN
I developed the project step by step: first the image classification model, and once it worked correctly, I built a full frame-by-frame video analysis pipeline. I added motion detection, bounding boxes, waste counting, and an informative visual panel. Finally, I integrated everything into two Streamlit apps so anyone can easily test the system by uploading their own images or videos.
