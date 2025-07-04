from click import option
import streamlit as st
import numpy as np
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from PIL import Image



# Título de la aplicación
imagen_encabezado = Image.open("images/logo.png")  
st.image(imagen_encabezado)
st.title("🩺 WarningSugar: Predicción Temprana de Diabetes con Big Data")
# Menú lateral
opcion_lateral = st.sidebar.selectbox("Navegación", ["Inicio", "Carga de Datos", "Pre procesamiento","Visualizacion", "Modelado","Predicción"])

# Contenido según la opción seleccionada
#Inicio
if opcion_lateral == "Inicio":
    resumen= "WarningSugar es una solución innovadora que busca prevenir la diabetes en adultos jóvenes de 20 a 25 años, utilizando tecnologías avanzadas de Big Data y Machine Learning. Nuestro proyecto combina análisis de datos clínicos, algoritmos predictivos y visualización interactiva para ofrecer una herramienta accesible y útil tanto para profesionales de la salud como para la población en general."
    st.write(resumen)
    st.header("**Objetivo**")
    st.write("*️⃣Prevenir la tendencia a la diabetes en adultos jóvenes en el rango de edad de 20 a 25 años a través del análisis de casos recientes, exámenes de sangre y modelo predictivo para mostrar mediante  una interfaz web los resultados y recomendaciones ")
    ##st.subheader("Integrantes:")
    st.header("Descripción del Dataset")
    descripcion_variables=pd.read_csv('csv/descripcion_variables.csv')
    st.dataframe(descripcion_variables)
    st.write("Dataset: https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset")
    st.markdown("## 👥 Integrantes")
    integrantes = [
        {
            "nombre": "Diego Josue Mendez Peralta",
            "correo": "diego.mendez.est@tecazuay.edu.ec",
            "genero": "Masculino",
            "edad": 19,
            "aporte": "Creación de la página web."
        },
        {
            "nombre": "Maria José Peña Carrera",
            "correo": "maria.pena.est@tecazuay.edu.ec",
            "genero": "Femenino",
            "edad": 32,
            "aporte": "Modelado de las gráficas."
        },
        {
            "nombre": "Jonnathan Fernando Tigre Bueno",
            "correo": "jonnathanf.tigre.est@tecazuay.edu.ec",
            "genero": "Masculino",
            "edad": 28,
            "aporte": "Análisis de datos."
        }
    ]
    # Distribuir en filas de 3 columnas máximo por fila (puedes ajustar según el diseño)
    cols = st.columns(len(integrantes))  # una columna por integrante

    for i, integrante in enumerate(integrantes):
        with cols[i]:
            st.markdown("----")
            st.markdown(f"**Nombre:** {integrante['nombre']}")
            st.markdown(f"**Correo:** {integrante['correo']}")
            st.markdown(f"**Género:** {integrante['genero']}")
            st.markdown(f"**Edad:** {integrante['edad']}")
            st.markdown(f"**Aporte:** {integrante['aporte']}")
# Carga de Archivos
elif opcion_lateral == "Carga de Datos":
    # Título
    st.title("⌛ Carga de Datos")
    # Cargar archivo
    ##archivo = st.file_uploader("📁 Sube el archivo CSV", type=["csv"]) 
    archivo= ('csv/diabetes_dataset.csv')
    # proceso de cargar los datos dentro de una condicion 
    if archivo is not None:
        # Cargar archivo y guardar la sesion
        data = pd.read_csv(archivo)
        data['Outcome'] = ((data['Fasting_Blood_Glucose'] >= 126) | (data['HbA1c'] > 6.5)).astype('int64')
        st.subheader("📌 Vista previa de los datos")
        data.drop(columns=['Unnamed: 0'], inplace=True)
        st.session_state.data = data 
        # Mostrar los primeros 5 registros
        st.write(data.head())
        # Info básica
        st.subheader("🔍 Información del DataFrame")
        st.write("Número de filas:", data.shape[0])
        st.write("Número de columnas:", data.shape[1])
        st.write("Encabezados", data.columns)
        st.write("Tipos de datos", data.dtypes)
        st.write("Estadisticas Generales", data.describe())
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.write("Informaciion DataFrame:")
        st.text(info_str)
        # Verificar valores nulos
        st.subheader("🧪 Valores nulos")
        st.write(data.isnull().sum())
        st.subheader("🧪 Porcentaje de Valores Nulos")
        st.write(data.isnull().sum()/ len(data)*100)
        # Verificar valores nulos
        st.subheader("🧪 Valores Duplicados")
        st.write(data.duplicated().sum())
        # verificar valores faltantes
        st.subheader("🧪 Valores Incompletos")
        st.write(data.isnull().any())
        # Verficar los valores unicos 
        st.subheader("🧪 Valores Unicos")
        st.write(data.apply(lambda x: len(x.unique())))
    else:
        st.info("⬆️ Por favor, sube un archivo CSV para comenzar.")
elif opcion_lateral == "Pre procesamiento":
    st.title("🔄 Pre Procesamiento de Datos")

    if 'data' in st.session_state:
        data = st.session_state.data
        st.subheader("⌛ Remplazando Valores Vacios")
        # remplazo de valores por la moda en la columna de consumo de alcohol
        most_common = data['Alcohol_Consumption'].mode()[0]
        data['Alcohol_Consumption'].fillna(most_common, inplace=True)
        st.write("Alcohol_Consumption reemplazados por la moda:", most_common)
        # visualizacion de los datos cambiados
        st.write(data['Alcohol_Consumption'].head())
        st.subheader("🆙 Edades")
        # Tabla de las edades
        st.write(data['Age'].value_counts())
        # agrupamiento de las edades
        st.subheader("Agrupando por rangos de edad")
        bins = [0, 19, 29, 39, 49, 59, 69, 79, float('inf')]
        labels = ['-20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=True)
        st.write(data['Age Group'])
        # Visualizacion de los datos limpios
        st.subheader("🧪 Datos Limpios")
        st.write(data.head())
        # Create a Matrix Plot
        st.subheader("📊 Mapa de Valores Completos")
        plt.figure(figsize=(10, 6))
        msno.matrix(data)
        plt.xticks(rotation=90)
        st.pyplot(plt)

        # Guardamos nuevamente en session_state
        st.session_state.data = data

        st.success("Preprocesamiento completado con éxito.")
    else:
        st.warning("⚠️ Primero debes cargar los datos en la sección 'Carga de Datos'.")
elif opcion_lateral == "Visualizacion":
    st.title("📊 Visualización de Datos")
    if 'data' in st.session_state:
        data = st.session_state.data
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        # Visualización de datos
        # Graficos Nicho de Clases de dsitribucion
        st.subheader("📊 Gráficos de Variables Numéricas")
        for col in numeric_cols:
            st.write(f"### {col}")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data[col], kde=True, ax=ax)
            ax.set_title(f"Distribución de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)
        st.subheader("📊 Gráficos de Variables Categóricas")
        for col in categorical_cols:
            st.write(f"### {col}")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=col, data=data, ax=ax)
            ax.set_title(f"Distribución de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
        # Visualización de correlaciones
        st.subheader("🔗 Mapa de Correlación")
        # Calcular la matriz de correlación
        corr_matrix = data[numeric_cols].corr()
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Matriz de Correlación entre Variables Numéricas")
        st.pyplot(fig)
        # diabetes por grupos de edad
        st.subheader("📊 Tasa de Diabetes por Grupos de Edad")
        bins = range(0, 101, 10)
        labels = [f'{i}-{i+9}' for i in bins[:-1]]
        data['age_group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
        age_group_diabetes = data.groupby('age_group', observed=True)['Outcome'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='age_group', y='Outcome', data=age_group_diabetes, marker='o', ax=ax)
        ax.set_title('Diabetes Rate by Age Group')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Proportion with Diabetes')
        ax.set_ylim(0, 1)
        ax.grid(True)
        st.pyplot(fig)
        # Casos de Diabetes
        st.subheader("📊 Clase Objetivo ")
        fig, ax = plt.subplots()
        sns.countplot(x='Outcome', data=data, ax=ax)
        ax.set_title("Distribución de la Clase Objetivo (Outcome)")
        ax.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
        ax.set_ylabel("Frecuencia")
        ax.grid(True)
        st.pyplot(fig)
        # Relacion entre BMi y Outcome
        st.subheader("📊 Relación entre BMI y Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Outcome', y='BMI', data=data, palette='Set2', ax=ax)
        ax.set_title('Relación entre BMI y Diabetes')
        ax.set_xlabel('Diabetes (0 = No, 1 = Sí)')
        ax.set_ylabel('BMI')
        ax.grid(True)
        st.pyplot(fig)
        # Relacion entre Edad y Outcome
        st.subheader("📊 Relación entre Edad y Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Outcome', y='Age', data=data, palette='Set2', ax=ax)
        ax.set_title('Relación entre Edad y Diabetes')
        ax.set_xlabel('Diabetes (0 = No, 1 = Sí)')
        ax.set_ylabel('Edad')
        ax.grid(True)
        st.pyplot(fig)
        # Relacion entre HbA1c y Outcome
        st.subheader("📊 Relación entre HbA1c y Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Outcome', y='HbA1c', data=data, palette='Set2', ax=ax)
        ax.set_title('Relación entre HbA1c y Diabetes')
        ax.set_xlabel('Diabetes (0 = No, 1 = Sí)')
        ax.set_ylabel('HbA1c')
        ax.grid(True)
        st.pyplot(fig)
        # Relacion entre Fasting_Blood_Glucose y Outcome
        st.subheader("📊 Relación entre Fasting Blood Glucose y Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Outcome', y='Fasting_Blood_Glucose', data=data, palette='Set2', ax=ax)
        ax.set_title('Relación entre Fasting Blood Glucose y Diabetes')
        ax.set_xlabel('Diabetes (0 = No, 1 = Sí)')
        ax.set_ylabel('Fasting Blood Glucose')
        ax.grid(True)
        st.pyplot(fig)
        # Relacion entre Alcohol_Consumption y Outcome
        st.subheader("📊 Relación entre Alcohol Consumption y Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Alcohol_Consumption', hue='Outcome', data=data, palette='Set2', ax=ax)
        ax.set_title('Relación entre Alcohol Consumption y Diabetes')
        ax.set_xlabel('Alcohol Consumption')
        ax.set_ylabel('Frecuencia')
        ax.grid(True)
        st.pyplot(fig)
        # Relacion entre Smoking_Status y Outcome
        st.subheader("📊 Relación entre Smoking Status y Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Smoking_Status', hue='Outcome', data=data, palette='Set2', ax=ax)
        ax.set_title('Relación entre Smoking Status y Diabetes')
        ax.set_xlabel('Smoking Status')
        ax.set_ylabel('Frecuencia')
        ax.grid(True)
        st.pyplot(fig)
        # Relacion entre Blood_Pressure y Outcome
        st.subheader("📊 Relación entre Blood Pressure y Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Outcome', y='Blood_Pressure_Systolic', data=data, palette='Set2', ax=ax)
        ax.set_title('Relación entre Blood Pressure y Diabetes')
        ax.set_xlabel('Diabetes (0 = No, 1 = Sí)')
        ax.set_ylabel('Blood Pressure')
        ax.grid(True)
        st.pyplot(fig)
        data = st.session_state.data
        st.session_state.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
elif opcion_lateral == "Modelado":
    st.title("🤖 Modelado de Datos")
    if 'data' in st.session_state:
        data = st.session_state.data.copy()  
        categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        st.subheader("🔄 Preprocesamiento para el Modelado")
        # Codifica variables categóricas con LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col].astype(str))
        # Dividir características y etiqueta
        y = data['Outcome']
        X = data.drop('Outcome', axis=1)

        # Crear el modelo XGBoost
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        # División de datos
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

        # Escalado
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar modelo
        model.fit(X_train_scaled, y_train)

        # Evaluar en validación
        y_val_pred = model.predict(X_val_scaled)
        st.subheader("🔍 Resultados en VALIDACIÓN (10%)")
        st.write(f"Accuracy: {model.score(X_val_scaled, y_val) * 100:.2f}%")
        st.text(classification_report(y_val, y_val_pred))

        # Evaluar en test
        y_test_pred = model.predict(X_test_scaled)
        st.subheader("🧪 Resultados en TEST (10%)")
        st.write(f"Accuracy: {model.score(X_test_scaled, y_test) * 100:.2f}%")
        st.text(classification_report(y_test, y_test_pred))

        # Validación cruzada en todo el dataset escalado
        X_all_scaled = scaler.fit_transform(X)
        acc_scores = cross_val_score(model, X_all_scaled, y, cv=5)
        st.subheader("🔁 Validación Cruzada")
        st.write(f"Accuracy Promedio: {np.mean(acc_scores) * 100:.2f}%")

        f1_scores = cross_val_score(model, X_all_scaled, y, cv=5, scoring='f1_macro')
        st.write(f"F1 Macro Score Promedio: {np.mean(f1_scores):.4f}")
        joblib.dump(model, 'xgb_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(le, 'label_encoder.pkl')
    else:
        st.warning("⚠️ Primero debes cargar los datos en la sección 'Carga de Datos'.")
elif opcion_lateral == "Predicción":
    st.subheader("Predicción de riesgo de diabetes")

    # Cargar modelo y transformadores
    modelo = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    ohe = joblib.load("label_encoder.pkl")
    '''''
    # Formulario de entrada
    st.markdown("### Ingrese los datos del paciente:")

    edad = st.number_input("Edad", min_value=1, max_value=120, value=30)
    sexo = st.selectbox("Género", ["Male", "Female"])
    peso = st.number_input("Peso (kg)", min_value=30, max_value=200, value=70)
    altura = st.number_input("Altura (cm)", min_value=50, max_value=250, value=170)
    # Calcular BMI
    bmi = peso / ((altura / 100) ** 2)
    m_cintura = st.number_input("Medida de la cintura (cm)", min_value=50, max_value=150, value=80)
    m_glucosa = st.number_input("Nivel de glucosa (mg/dL)", min_value=50, max_value=300, value=100)
    HbA1c = st.number_input("Nivel de HbA1c (%)", min_value=4.0, max_value=15.0, value=5.5)
    p_arterial_sistolica = st.number_input("Presión arterial sistólica (mmHg)", min_value=80, max_value=200, value=120)
    p_arterial_diastolica = st.number_input("Presión arterial diastólica (mmHg)", min_value=50, max_value=150, value=80)
    colesterol_total = st.number_input("Colesterol total (mg/dL)", min_value=100, max_value=400, value=200)
    hdl = st.number_input("Colesterol HDL (mg/dL)", min_value=20, max_value=100, value=50)
    ldl = st.number_input("Colesterol LDL (mg/dL)", min_value=50, max_value=300, value=100)
    ggt = st.number_input("Gamma-glutamil transferasa (U/L)", min_value=10, max_value=200, value=30)
    acido_urico = st.number_input("Ácido úrico (mg/dL)", min_value=2.0, max_value=10.0, value=5.0)
    grado_actividad = st.selectbox("Grado de actividad física", ["Sedentario", "Ligero", "Moderado", "Intenso"])
    consumo_calorias = st.number_input("Consumo calórico diario (kcal)", min_value=1000, max_value=5000, value=2000)
    consumo_alcohol = st.selectbox("Consumo de alcohol", ["Nunca", "Ocasional", "Regular", "Excesivo"])
    fumador = st.selectbox("Estado de fumador", ["Nunca", "Ex-fumador", "Fumador actual"])
    ante_familiares_diabetes = st.selectbox("Antecedentes familiares de diabetes", ["Sí", "No"])
    hipertension = st.selectbox("Hipertensión", ["Sí", "No"])

    # Construir dataframe con un solo registro
    datos_nuevos = pd.DataFrame({
        "Age": [edad],
        "Sex": [sexo],
        "hypertension": [hipertension],
        "heart_disease": [enfermedad_cardiaca],
        "ever_married": [estado_casado],
        "work_type": [tipo_trabajo],
        "Residence_type": [residencia],
        "avg_glucose_level": [nivel_glucosa],
        "bmi": [bmi],
        "smoking_status": [fumador]
    })

    # Separar columnas categóricas
    columnas_cat = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    columnas_num = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

    # Codificar y escalar
    datos_cat = ohe.transform(datos_nuevos[columnas_cat])
    datos_num = scaler.transform(datos_nuevos[columnas_num])

    datos_procesados = np.concatenate([datos_num, datos_cat], axis=1)

    # Hacer predicción
    prediccion = modelo.predict(datos_procesados)
    probabilidad = modelo.predict_proba(datos_procesados)[0][1]

    if prediccion[0] == 1:
        st.error(f"🔴 Alto riesgo de diabetes. Probabilidad: {probabilidad:.2f}")
    else:
        st.success(f"🟢 Bajo riesgo de diabetes. Probabilidad: {probabilidad:.2f}")
'''