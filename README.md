# 🧬 Predicción de Diabetes con Machine Learning

Proyecto de Data Science aplicado a datos médicos para la predicción de riesgo de diabetes utilizando modelos de machine learning.

## 🚀 Tecnologías

- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn

## 📊 Dataset

Pima Indians Diabetes Dataset (datos médicos reales de pacientes)

## 🔍 Proceso

### 1. Análisis exploratorio (EDA)
- Estadísticas descriptivas
- Distribución de la variable objetivo
- Identificación de desbalance en el dataset

### 2. Limpieza de datos
- Detección de valores inválidos (0 en variables médicas)
- Conversión a valores nulos (NaN)
- Imputación usando la mediana

### 3. Modelado
- Regresión logística
- Evaluación con métricas (accuracy, recall)

### 4. Interpretación
- Análisis de importancia de variables
- Identificación de factores clave (glucosa, BMI, edad, genética)

## 📈 Resultados

- Accuracy ~ 75-80%
- Identificación de variables relevantes en la predicción
- Modelo interpretable aplicado a datos médicos

## 🧠 Conclusiones

El modelo muestra que factores como la glucosa, el índice de masa corporal (BMI) y la predisposición genética son determinantes en la predicción de diabetes.

## ⚙️ Cómo ejecutar

```bash
pip install -r requirements.txt
python main.py
