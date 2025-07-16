import requests  
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image

url = "https://jsonplaceholder.typicode.com/users"
## se crea una varible para guardar
response = requests.get(url)
## verificamos si la solicitud fue exitosa
if response.status_code == 200:
    users = response.json()
else:
    print("Error al consumir la API")
    exit()
## se guarda en una base datos SQLite
conn = sqlite3.connect('usuarios2.db')
## conn permite conocer la conexion a la base de datos
## se crea la extructura de la  tabla 
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    username TEXT,
    email TEXT,
    phone TEXT,
    website TEXT
)
''')
for user in users:
    cursor.execute('''
    INSERT OR REPLACE INTO users (id, name, username, email, phone, website)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        user['id'],
        user['name'],
        user['username'],
        user['email'],
        user['phone'],
        user['website']
    ))
conn.commit()
conn.close()

conn = sqlite3.connect('usuarios.db')
df = pd.read_sql_query("SELECT * FROM users", conn)
conn.close()

# Título de la aplicación
imagen_encabezado = Image.open("images/logo.png")  
st.image(imagen_encabezado)
st.title("Practica Json,SQLite y DataFrame")
# visaulizar los datos cargados
st.subheader("🔍 Información del DataFrame")
st.write("Datos cargados desde la API y guardados en SQLite:")
st.dataframe(df)
st.write("Cantidad de usuarios:", len(df))
st.write("Número de filas:", df.shape[0])
st.write("Número de columnas:", df.shape[1])
st.write("Encabezados", df.columns)
st.write("Tipos de datos:", df.dtypes)
st.write("Descripción estadística:", df.describe())

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# grafico de la cantidad de caracteres en los nombres
st.subheader("📊 Visualización de Datos")
st.subheader("Gráfico distribución de los nombres")
df['name_length'] = df['name'].apply(len)
fig, ax = plt.subplots(figsize=(10, 6)) 
sns.histplot(df['name_length'], bins=10, kde=True, ax=ax)
ax.set_title('Distribución del número de caracteres en los nombres')
ax.set_xlabel('Cantidad de caracteres')
ax.set_ylabel('Frecuencia')
ax.grid(True)
st.pyplot(fig)

#Conteo de usuarios por dominio de correo 
df['email_domain'] = df['email'].apply(lambda x: x.split('@')[-1])
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, y='email_domain', order=df['email_domain'].value_counts().index)
ax1.set_title('Conteo de usuarios por dominio de correo')
ax1.set_xlabel('Cantidad de usuarios')
ax1.set_ylabel('Dominio de correo')
st.pyplot(fig)

# grafico dominios web
df['website_domain'] = df['website'].apply(lambda x: x.split('.')[-1])
fig, ax2 = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, y='website_domain', order=df['website_domain'].value_counts().index)
ax2.set_title('Conteo de usuarios por dominio de sitio web')
ax2.set_xlabel('Cantidad de usuarios')
ax2.set_ylabel('Dominio de sitio web')
st.pyplot(fig)
