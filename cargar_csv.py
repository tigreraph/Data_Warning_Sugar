from turtle import st
import pandas as pd
import psycopg2
from psycopg2 import sql

# Cargar el CSV y renombrar columnas a minúsculas
df = pd.read_csv("dataset_diabetes_modificado(outcome).csv")
df.columns = [col.lower() for col in df.columns]

# Conexión a la base de datos
def conectar_db():
    return psycopg2.connect(st.secrets["connections"]["DB_URL"])

# Función para insertar los datos en la tabla
def insertar_csv_en_postgres(df):
    conn = conectar_db()
    cursor = conn.cursor()

    for _, fila in df.iterrows():
        cursor.execute("""
            INSERT INTO formulario_respuestas (
                age, sex, ethnicity, peso, altura, bmi, waist_circumference,
                blood_pressure_systolic, blood_pressure_diastolic,
                physical_activity_level, alcohol_consumption, smoking_status,
                family_history_of_diabetes, previous_gestational_diabetes, outcome
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            fila["age"], fila["sex"], fila["ethnicity"], fila["peso"], fila["altura"],
            fila["bmi"], fila["waist_circumference"], fila["blood_pressure_systolic"],
            fila["blood_pressure_diastolic"], fila["physical_activity_level"],
            fila["alcohol_consumption"], fila["smoking_status"],
            bool(fila["family_history_of_diabetes"]),
            bool(fila["previous_gestational_diabetes"]),
            int(fila["outcome"])
        ))

    conn.commit()
    conn.close()
    st.success("✅ CSV insertado correctamente en la base de datos.")

# Ejecutar con botón en Streamlit
if st.button("📂 Insertar CSV a la base de datos"):
    insertar_csv_en_postgres(df)
