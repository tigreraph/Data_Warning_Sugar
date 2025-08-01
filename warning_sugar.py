from click import option
from openpyxl import load_workbook
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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score
from PIL import Image
import os
import psycopg2
from psycopg2 import sql


# conexion a la base de datos
def conectar_db():
    return psycopg2.connect(st.secrets["connections"]["DB_URL"])
# crear tabla si no existe
def crear_tabla_si_no_existe():
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS formulario_respuestas (
            id SERIAL PRIMARY KEY,
            age INTEGER,
            sex VARCHAR(50),
            ethnicity VARCHAR(50),
            peso FLOAT,
            altura FLOAT,
            bmi FLOAT,
            waist_circumference FLOAT,
            blood_pressure_systolic FLOAT,
            blood_pressure_diastolic FLOAT,
            physical_activity_level VARCHAR(50),
            alcohol_consumption VARCHAR(50),
            smoking_status VARCHAR(50),
            family_history_of_diabetes BOOLEAN,
            previous_gestational_diabetes BOOLEAN,
            outcome INTEGER
        );
    """)
    conn.commit()
    conn.close()

def asegurar_tabla():
    if "tabla_creada" not in st.session_state:
        try:
            crear_tabla_si_no_existe()
            st.session_state["tabla_creada"] = True
        except Exception as e:
            st.error(f"❌ Error al crear/verificar la tabla: {e}")
# Mostrar registros guardados desde la base de datos (después de predicción)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def mostrar_registros_guardados():
    try:
        conn = conectar_db()
        df = pd.read_sql("SELECT * FROM formulario_respuestas ORDER BY id DESC", conn)
        
        st.subheader("📌 Vista previa de los datos")
        columns_drops = ["peso", "altura"]
        df = df.drop(columns=columns_drops, errors='ignore')  # Evita error si no existen
        st.dataframe(df.head(10))

        st.subheader("🔍 Información del DataFrame")
        st.write("Número de filas:", df.shape[0])
        st.write("Número de columnas:", df.shape[1])

        st.subheader("Tipos de datos")
        st.write(df.dtypes)

        st.subheader("Estadísticas Generales")
        st.write(df.describe())

        # Crear columna de grupo de edad
        bins = [0, 19, 29, 39, 49, 59, 69, 79, float('inf')]
        labels = ['-20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

        # Gráfico de conteo por grupo de edad
        st.subheader("📊 Distribución por grupo de edad")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.countplot(x='age_group', data=df, palette='Blues_d', ax=ax1)
        ax1.set_title("Distribución de registros por grupo de edad", fontsize=14)
        ax1.set_xlabel("Grupo de Edad")
        ax1.set_ylabel("Cantidad")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True)
        st.pyplot(fig1)

        # Proporción de diabetes por grupo de edad
        age_group_diabetes = df.groupby('age_group')['outcome'].mean().reset_index()
        st.subheader("📊 Porcentaje de diabetes por grupo de edad")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='age_group', y='outcome', data=age_group_diabetes, marker='o', ax=ax2)
        ax2.set_title('Proporción de casos de diabetes por grupo de edad')
        ax2.set_xlabel('Grupo de Edad')
        ax2.set_ylabel('Proporción con Diabetes')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        st.pyplot(fig2)

        # Casos de diabetes
        st.subheader("📊 Frecuencia de casos de diabetes")
        fig3, ax3 = plt.subplots()
        sns.countplot(x='outcome', data=df, ax=ax3)
        ax3.set_title("Casos de diabetes")
        ax3.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
        ax3.set_ylabel("Frecuencia")
        ax3.grid(True)
        st.pyplot(fig3)

        # Relación entre BMI y Outcome
        st.subheader("📊 Relación entre BMI y Diabetes")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='outcome', y='bmi', data=df, palette='Set2', ax=ax4)
        ax4.set_title('Relación entre BMI y Diabetes')
        ax4.set_xlabel('Diabetes (0 = No, 1 = Sí)')
        ax4.set_ylabel('BMI')
        ax4.grid(True)
        st.pyplot(fig4)

        # Relación entre Edad y Outcome
        st.subheader("📊 Relación entre Edad y Diabetes")
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='outcome', y='age', data=df, palette='Set2', ax=ax5)
        ax5.set_title('Relación entre Edad y Diabetes')
        ax5.set_xlabel('Diabetes (0 = No, 1 = Sí)')
        ax5.set_ylabel('Edad')
        ax5.grid(True)
        st.pyplot(fig5)

        conn.close()
    except Exception as e:
        st.error(f"❌ Error al cargar registros: {e}")



# --- Guardar datos del formulario en la base de datos 
def guardar_en_base_de_datos(form_data,outcome):
    asegurar_tabla()
    conn = conectar_db()
    cursor = conn.cursor()

    cursor.execute(
        sql.SQL("""
            INSERT INTO formulario_respuestas (
                age, sex, ethnicity, peso, altura, bmi, waist_circumference,
                blood_pressure_systolic, blood_pressure_diastolic, physical_activity_level,
                alcohol_consumption, smoking_status, family_history_of_diabetes, previous_gestational_diabetes,
                outcome
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """),
        (
            form_data.get("Age"),
            form_data.get("Sex"),
            form_data.get("Ethnicity"),
            form_data.get("Peso"),
            form_data.get("Altura"),
            form_data.get("BMI"),
            form_data.get("Waist_Circumference"),
            form_data.get("Blood_Pressure_Systolic"),
            form_data.get("Blood_Pressure_Diastolic"),
            form_data.get("Physical_Activity_Level"),
            form_data.get("Alcohol_Consumption"),
            form_data.get("Smoking_Status"),
            form_data.get("Family_History_of_Diabetes") in ["Sí", "Yes", "1", True],
            form_data.get("Previous_Gestational_Diabetes") in ["Sí", "Yes", "1", True],
            outcome
        )
    )
    conn.commit()
    conn.close()
def mostrar_categoria_riesgo(aux):
    probabilidad = (aux*100)
    categoria = ""
    if probabilidad <= 33:
        categoria = "BAJO"
    elif probabilidad <= 66:
        categoria = "MEDIO"
    else:
        categoria = "ALTO"

    st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <div style="background-color:#ecf0f1;border-radius:30px;display:flex;padding:10px;gap:10px;justify-content:space-between;max-width:700px;width:100%">
                <div style="flex: 1; text-align:center; background-color:{'#27ae60' if categoria == 'BAJO' else '#ecf0f1'};padding:10px 0;border-radius:25px;color:{'white' if categoria == 'BAJO' else '#888'};font-weight:bold">
                    BAJO<br><span style='font-size:12px'>1–33%</span>
                </div>
                <div style="flex: 1; text-align:center; background-color:{'#f1c40f' if categoria == 'MEDIO' else '#ecf0f1'};padding:10px 0;border-radius:25px;color:{'white' if categoria == 'MEDIO' else '#888'};font-weight:bold">
                    MEDIO<br><span style='font-size:12px'>34–66%</span>
                </div>
                <div style="flex: 1; text-align:center; background-color:{'#e74c3c' if categoria == 'ALTO' else '#ecf0f1'};padding:10px 0;border-radius:25px;color:{'white' if categoria == 'ALTO' else '#888'};font-weight:bold">
                    ALTO<br><span style='font-size:12px'>67–100%</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
def mostrar_recomendacion_riesgo(proba):
    porcentaje = proba * 100
    st.markdown("## ¿Qué hacer ahora?")
    if porcentaje < 34:
        st.success("Tienes un **nivel de riesgo bajo**. ¡Excelente! Sigue manteniendo un estilo de vida saludable con actividad física regular, alimentación balanceada y chequeos médicos periódicos.")
    elif porcentaje < 67:
        st.warning("Tienes un **nivel de riesgo moderado**. Aunque no es preocupante, es un buen momento para hacer pequeños cambios: mejorar tu alimentación, reducir el consumo de alcohol, dejar de fumar o aumentar tu actividad física.")
    else:
        st.error("Tienes un **nivel de riesgo alto**. Es muy importante que consultes con un profesional de la salud lo antes posible. Cambios urgentes en tu estilo de vida, como alimentación saludable, ejercicio y control médico son fundamentales.")
def mostrar_factores_modificables(datos):
    cintura = datos.get("Waist_Circumference", 0)
    imc = datos.get("BMI", 0)
    pas = datos.get("Blood_Pressure_Systolic", 0)
    pad = datos.get("Blood_Pressure_Diastolic", 0)

    st.markdown("### 🔧 Factores de riesgo que puedes mejorar")
    st.info("Incluso pequeños cambios pueden ayudarte a reducir tu riesgo. A continuación, te mostramos algunas recomendaciones según tus respuestas:")

    if cintura >= 90:
        st.warning(f"📏 Circunferencia de cintura: **{cintura} cm**")
        if cintura < 100:
            st.markdown("- Estás en un rango ligeramente elevado. Considera bajar tu consumo de grasas y azúcares simples.")
        elif cintura < 110:
            st.markdown("- Cintura en rango moderadamente alto. Aumentar la actividad física y cuidar porciones es clave.")
        else:
            st.markdown("- Alto riesgo por grasa abdominal. Consulta con nutricionista para plan personalizado.")

    if imc >= 25:
        st.warning(f"⚖️ IMC: **{imc}**")
        if imc < 30:
            st.markdown("- Estás en sobrepeso. Baja de peso entre 5-10% puede reducir mucho tu riesgo.")
        else:
            st.markdown("- Obesidad. Acude a control médico y nutricional urgente para prevención.")

    if pas >= 130 or pad >= 85:
        st.warning(f"❤️ Presión arterial: **{pas}/{pad} mmHg**")
        st.markdown("- Considera reducir el consumo de sal, manejar el estrés y hacer actividad física regularmente.")

    if cintura < 90 and imc < 25 and pas < 130 and pad < 85:
        st.success("🎉 ¡Excelente! Eres una persona saludable.")
def mostrar_creditos_proyecto():
    st.markdown("---")
    st.header("🔬 Acerca del Proyecto WarningSugar")

    resumen = (
        "WarningSugar es una solución innovadora que busca prevenir la diabetes en personas, utilizando tecnologías avanzadas de Big Data y Machine Learning. Nuestro "
        "proyecto combina análisis de datos clínicos, algoritmos predictivos y visualización interactiva "
        "para ofrecer una herramienta accesible y útil tanto para profesionales de la salud como para la población en general."
    )
    st.write(resumen)

    st.header("🎯 Objetivo")
    st.write(
        "Prevenir la tendencia a la diabetes en adultos a través "
        "del análisis de casos recientes, exámenes de sangre y modelo predictivo para mostrar mediante una interfaz web los resultados y recomendaciones."
    )

    st.header("📊 Descripción del Dataset")
    try:
        descripcion_variables = pd.read_csv("csv/descripcion_variables.csv")
        st.dataframe(descripcion_variables)
    except Exception as e:
        st.error(f"No se pudo cargar el archivo CSV: {e}")

    st.markdown("📁 [Link al Dataset en Kaggle](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset)")

    st.markdown("## 👥 Integrantes")
    integrantes = [
        {
            "nombre": "Diego Josue Mendez Peralta",
            "correo": "diego.mendez.est@tecazuay.edu.ec",
            "genero": "Masculino",
            "edad": 19,
            "aporte": "Análisis de datos"
        },
        {
            "nombre": "Jonnathan Fernando Tigre Bueno",
            "correo": "jonnathanf.tigre.est@tecazuay.edu.ec",
            "genero": "Masculino",
            "edad": 28,
            "aporte": "Creación de la interfaz web y modelo predictivo"
        }
    ]

    cols = st.columns(len(integrantes))
    for i, integrante in enumerate(integrantes):
        with cols[i]:
            st.markdown("----")
            st.markdown(f"**👤 Nombre:** {integrante['nombre']}")
            st.markdown(f"📧 **Correo:** {integrante['correo']}")
            st.markdown(f"⚧️ **Género:** {integrante['genero']}")
            st.markdown(f"🎂 **Edad:** {integrante['edad']}")
            st.markdown(f"💡 **Aporte:** {integrante['aporte']}")

# Título de la aplicación
imagen_encabezado = Image.open("images/logo.png")  
st.image(imagen_encabezado)
st.title(" Tecnología Superior en Big Data")
st.title("🩺 WarningSugar: Predicción Temprana de Diabetes")
# Menú lateral
##opcion_lateral = st.sidebar.selectbox("Navegación", ['Formulario',"Presentación", "Carga de Datos", "Pre procesamiento","Visualizacion", "Modelado"])
#Formulario 

# Contenido según la opción seleccionada
opcion_lateral = "Formulario"
if opcion_lateral == "Formulario":
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "form_data" not in st.session_state:
        st.session_state.form_data = {}

    def next_step():
        if st.session_state.step < total_pasos - 1:
            st.session_state.step += 1

    def prev_step():
        if isinstance(st.session_state.step, int) and st.session_state.step > 0:
            st.session_state.step -= 1

    # Estilo visual
    st.markdown("""
        <style>
        button[kind="secondary"] {
            background-color: #002D72;
            color: white;
            border-radius: 25px;
            padding: 0.75em 1.5em;
            font-size: 16px;
            margin: 4px;
        }
        button[kind="secondary"]:hover {
            background-color: #0052CC;
            color: white;
        }
        .pregunta-formulario {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    preguntas = [
        ("Age", "number", {"label": "¿Cuál es tu edad?", "min_value": 0, "max_value": 120}, "images/edad.png"),
        ("Sex", "select", {"label": "¿Cuál es tu género?", "options": ["Masculino", "Femenino"]}, "images/sexo.png"),
        ("Ethnicity", "select", {"label": "¿Cuál es tu etnia?", "options": ["Caucásico", "Asiático", "Hispano", "Afrodescendiente"]}, "images/etnia.png"),
        ("PesoAltura", "peso_altura", {"label": "Ingresa tu peso y altura para calcular el IMC"}, "images/imc.png"),
        ("Waist_Circumference", "number", {"label": "¿Tu cintura (cm)?", "min_value": 0.0, "format": "%.2f"}, "images/cintura.png"),
        ("Blood_Pressure_Systolic", "number", {"label": "¿Presión sistólica (mmHg)?", "min_value": 0.0, "format": "%.2f"}, "images/pas.png"),
        ("Blood_Pressure_Diastolic", "number", {"label": "¿Presión diastólica (mmHg)?", "min_value": 0.0, "format": "%.2f"}, "images/pad.png"),
        ("Physical_Activity_Level", "select", {"label": "¿Nivel de actividad física?", "options": ["Baja", "Moderada", "Alta"]}, "images/actividad.png"),
        ("Alcohol_Consumption", "select", {"label": "¿Con qué frecuencia consumes alcohol?", "options": ["Moderado", "Alto"]}, "images/alcohol.png"),
        ("Smoking_Status", "select", {"label": "¿Con qué frecuencia fumas?", "options": ["No fumo", "Exfumador", "Fumador"]}, "images/fumar.png"),
        ("Family_History_of_Diabetes", "select", {"label": "¿Antecedentes familiares de diabetes?", "options": ["Sí", "No"]}, "images/familia.png"),
        ("Previous_Gestational_Diabetes", "select", {"label": "¿Tuviste diabetes gestacional?", "options": ["Sí", "No"]}, "images/gestacional.png")
    ]

    total_pasos = len(preguntas)

    # 🔁 TRADUCIR RESPUESTAS AL INGLÉS PARA EL MODELO
    def traducir_datos(form_data):
        mapeos = {
            "Sex": {"Masculino": "Male", "Femenino": "Female"},
            "Ethnicity": {
                "Caucásico": "White",
                "Asiático": "Asian",
                "Hispano": "Hispanic",
                "Afrodescendiente": "Black",
            },
            "Physical_Activity_Level": {"Baja": "Low", "Moderada": "Moderate", "Alta": "High"},
            "Alcohol_Consumption": {"Moderado": "Moderate", "Alto": "Heavy"},
            "Smoking_Status": {"No fumo": "Never", "Exfumador": "Former", "Fumador": "Current"},
            "Family_History_of_Diabetes": {"Sí": "1", "No": "0"},
            "Previous_Gestational_Diabetes": {"Sí": "1", "No": "0"}
        }
        traducido = form_data.copy()
        for clave, mapeo in mapeos.items():
            if clave in traducido:
                traducido[clave] = mapeo.get(traducido[clave], traducido[clave])
        return traducido

    # Mostrar resumen al final
    if st.session_state.step == "resumen":
        st.markdown("## 🧾 Resumen de tus respuestas:")
        datos = st.session_state.form_data

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🧍 Edad: `{datos.get('Age', '-')}` años")
            st.markdown(f"#### 🧬 Género: `{datos.get('Sex', '-')}`")
            st.markdown(f"#### 🌍 Etnia: `{datos.get('Ethnicity', '-')}`")
            st.markdown(f"#### ⚖️ Peso: `{datos.get('Peso', '-')}` kg")
            st.markdown(f"#### 📏 Altura: `{datos.get('Altura', '-')}` cm")
            st.markdown(f"#### 📉 IMC: `{datos.get('BMI', '-')}`")

        with col2:
            st.markdown(f"#### 📐 Circunferencia cintura: `{datos.get('Waist_Circumference', '-')}` cm")
            st.markdown(f"#### ❤️ PAS: `{datos.get('Blood_Pressure_Systolic', '-')}` mmHg")
            st.markdown(f"#### ❤️ PAD: `{datos.get('Blood_Pressure_Diastolic', '-')}` mmHg")
            st.markdown(f"#### 🏃 Actividad física: `{datos.get('Physical_Activity_Level', '-')}`")
            st.markdown(f"#### 🍷 Alcohol: `{datos.get('Alcohol_Consumption', '-')}`")
            st.markdown(f"#### 🚬 Fuma: `{datos.get('Smoking_Status', '-')}`")
            st.markdown(f"#### 👪 Antecedentes familiares: `{datos.get('Family_History_of_Diabetes', '-')}`")
            st.markdown(f"#### 🤰 Diabetes gestacional: `{datos.get('Previous_Gestational_Diabetes', '-')}`")

        
        # Boton predicción
        if "prediccion_realizada" not in st.session_state:
            if st.button("🔍 Predecir riesgo de diabetes"):
                # cargar modelo y codificadores
                modelo = joblib.load('rf_model.pkl')
                scaler = joblib.load("scaler.pkl")
                label_encoders = joblib.load('label_encoders.pkl')
                categorical_cols = joblib.load("categorical_cols.pkl")
                columnas_modelo = joblib.load('columnas_modelo.pkl')
                # Traducir datos del formulario
                datos_modelo = traducir_datos(st.session_state.form_data)
                X_nuevo = pd.DataFrame([datos_modelo])
                for col in categorical_cols:
                    if col in X_nuevo:
                        le = label_encoders[col]
                        X_nuevo[col] = le.transform(X_nuevo[col].astype(str))
                # Asegurar que las columnas coincidan con el modelo
                X_nuevo = X_nuevo[columnas_modelo]
                # Escalar los datos
                X_nuevo_scaled = scaler.transform(X_nuevo)
                # Realizar la predicción
                prediccion = modelo.predict(X_nuevo_scaled)[0]
                # Calcular la probabilidad de diabetes
                proba = modelo.predict_proba(X_nuevo_scaled)[0][1]
                try:
                    ## guardar en la base de datos el formulario y la predicción
                    guardar_en_base_de_datos(traducir_datos(datos_modelo), int(prediccion))
                except Exception as e:
                    st.error(f"❌ Error al guardar en la base de datos: {e}")
                # Mostrar resultados
                mostrar_categoria_riesgo(proba)
                st.subheader(f"📊 El Resultado de la predicción: {proba * 100:.2f}%")
                mostrar_recomendacion_riesgo(proba)
                mostrar_factores_modificables(st.session_state.form_data)
                # Guardar la probabilidad en session_state
                st.session_state["prediccion_realizada"] = True
                st.session_state["proba"] = proba
        else:
            # Mostrar resultado anterior sin volver a ejecutar la predicción
            mostrar_categoria_riesgo(st.session_state["proba"])
            st.write(" ")
            st.write(" ")
            st.write(f"📊 El Resultado de la predicción: {st.session_state['proba'] * 100:.2f}%")
            mostrar_recomendacion_riesgo(st.session_state["proba"])
            mostrar_factores_modificables(st.session_state.form_data)

        # Mostrar botón SIEMPRE que ya se haya predicho
        if "prediccion_realizada" in st.session_state:
            if st.button("📋 Ver análisis de registros guardados"):
                mostrar_registros_guardados()
                mostrar_creditos_proyecto() 
    # Continuar con preguntas paso a paso
if isinstance(st.session_state.get("step"), int):
    paso = st.session_state.step
    clave, tipo, kwargs, ruta_imagen = preguntas[paso]

    # Barra tipo wizard
    barra = ""
    for i in range(total_pasos):
        color = "#0047AB" if i == paso else "#ccc"
        texto = f"<span style='background:{color};color:white;border-radius:50%;padding:6px 12px;margin:3px'>{i+1}</span>"
        barra += texto
    st.markdown(f"<div style='text-align:center;font-size:20px'>{barra}</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    col_txt, col_img = st.columns([2, 1])
    with col_txt:
        st.markdown(f"<div class='pregunta-formulario'>{kwargs['label']}</div>", unsafe_allow_html=True)

        if tipo == "number":
            with st.form(key=f"form_{clave}"):
                valor_actual = st.session_state.form_data.get(clave, kwargs.get("min_value", 0))
                respuesta = st.number_input(label="", value=valor_actual, **{k: v for k, v in kwargs.items() if k not in ["label", "value"]})

                col1, col2 = st.columns([1, 1])
                with col1:
                    btn_prev = st.form_submit_button("⬅️ Anterior")
                with col2:
                    if paso < total_pasos - 1:
                        btn_next = st.form_submit_button("Siguiente ➡️")
                    else:
                        btn_next = st.form_submit_button("✅ Finalizar")

                if btn_prev and paso > 0:
                    st.session_state.form_data[clave] = respuesta
                    prev_step()
                    st.rerun()

                if btn_next:
                    st.session_state.form_data[clave] = respuesta
                    if paso < total_pasos - 1:
                        next_step()
                    else:
                        st.session_state.step = "resumen"
                    st.rerun()

        elif tipo == "select":
            opciones = kwargs["options"]
            cols = st.columns(3)
            for i, opcion in enumerate(opciones):
                with cols[i % 3]:
                    if st.button(opcion, key=f"{clave}_{i}"):
                        st.session_state.form_data[clave] = opcion
                        if paso < total_pasos - 1:
                            next_step()
                            st.rerun()
                        else:
                            st.session_state.step = "resumen"
                            st.rerun()

        elif tipo == "peso_altura":
            # Validación para el formulario de peso y altura
            peso = st.number_input("⚖️ Peso (kg)", min_value=0.01, format="%.2f")
            altura = st.number_input("📏 Altura (cm)", min_value=0.01, format="%.2f")

            # Calcular IMC si ambos valores son mayores a cero
            if peso > 0 and altura > 0:
                altura_m = altura / 100  # Convertir altura a metros
                imc = peso / (altura_m ** 2)  # Fórmula para calcular el IMC
                st.markdown(f"### 💡 Tu IMC es: `{imc:.2f}`")  # Mostrar IMC calculado

            # Validación: Si el usuario ingresa valores negativos o cero, mostrar advertencia
            if peso <= 0 or altura <= 0:
                st.warning("⚠️ Los valores de peso y altura deben ser positivos y mayores que cero.")

            # Botón para pasar al siguiente paso
            btn_next = st.button("Siguiente ➡️")

            if btn_next:
                # Si los valores son válidos, se guarda la información
                if peso > 0 and altura > 0:
                    st.session_state.form_data["Peso"] = peso
                    st.session_state.form_data["Altura"] = altura
                    st.session_state.form_data["BMI"] = round(imc, 2)  # Guardamos el IMC calculado
                    next_step()
                    st.rerun()
                else:
                    st.warning("⚠️ Completa correctamente los campos de peso y altura antes de continuar.")

    with col_img:
        try:
            st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
            st.image(ruta_imagen, width=220)
            st.markdown("</div>", unsafe_allow_html=True)
        except:
            st.info("Imagen no disponible.")

#Presentación


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
