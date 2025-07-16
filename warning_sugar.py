import streamlit as st
import numpy as np
import pandas as pd
import io
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import LabelEncoder
import missingno as msno

# Título de la aplicación
imagen_encabezado = Image.open("images/logo.png")  
st.image(imagen_encabezado)
st.title("🩺 WarningSugar: Predicción Temprana de Diabetes con Big Data")
# Menú lateral
opcion_lateral = st.sidebar.selectbox("Navegación", ["Inicio", "Carga de Datos", "Pre procesamiento","Visualizacion", "Modelado","Pruebas"])

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
        # Tomar una muestra de 100 registros para mejor visualización
        muestra = data.sample(n=100, random_state=42)

        # Ordenar los datos por índice para una mejor visualización
        muestra_ordenada = muestra.sort_index()

        # Crear el gráfico con dos ejes Y
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Configurar el primer eje Y para BMI
        color1 = 'tab:blue'
        ax1.set_xlabel('Número de Muestra')
        ax1.set_ylabel('BMI', color=color1)
        line1 = ax1.plot(range(len(muestra_ordenada)), muestra_ordenada['BMI'], color=color1, label='BMI')
        ax1.tick_params(axis='y', labelcolor=color1)

        # Crear el segundo eje Y para la presión arterial
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Presión Arterial Sistólica', color=color2)
        line2 = ax2.plot(range(len(muestra_ordenada)), muestra_ordenada['Blood_Pressure_Systolic'], color=color2, label='Presión Arterial')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Añadir título y leyenda
        plt.title('BMI y Presión Arterial Sistólica por Muestra')

        # Combinar las líneas para la leyenda
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        # Ajustar el diseño para evitar superposición
        plt.tight_layout()

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
    # Visualización de datos
