import streamlit as st
import pandas as pd

st.title("Vista de descripción de variables")

# Verifica que el archivo esté en el mismo directorio
try:
    descripcion_variables = pd.read_csv('descripcion_variables.csv')
    st.write(descripcion_variables.head(22))
except FileNotFoundError:
    st.error("No se encontró el archivo 'descripcion_variables.csv'.")