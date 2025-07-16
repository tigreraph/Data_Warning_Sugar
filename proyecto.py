import streamlit as st

# Título de la aplicación
st.title("Marca")

# Menú lateral
opcion_lateral = st.sidebar.selectbox("Navegación", ["Inicio", "Carga de Datos", "Pre procesamiento", "Modelodo","Pruebas"])

# Contenido según la opción seleccionada
if opcion_lateral == "Inicio":
    st.header("Bienvenido a la Página Principal")
    st.write("Aquí puedes navegar a diferentes actividades.")
    
elif opcion_lateral == "Carga de Datos":
    st.header("Actividad 1")
    st.write("Descripción de la actividad 1.")
    # Aquí puedes agregar más contenido o funcionalidades específicas para la actividad 1

elif opcion_lateral == "Pre procesamiento":
    st.header("Pre procesamiento")
    st.write("Descripción de la actividad 2.")
    # Aquí puedes agregar más contenido o funcionalidades específicas para la actividad 2

elif opcion_lateral == "Modelodo":
    st.header("Modelado")
    st.write("Descripción de la actividad 3.")
    # Aquí puedes agregar más contenido o funcionalidades específicas para la actividad 3

elif opcion_lateral == "Pruebas":
    st.header("Pruebas")
    st.write("Descripción de la actividad 3.")
    # Aquí puedes agregar más contenido o funcionalidades específicas para la actividad 3

# Expander para más información
with st.expander("Ver más"):
    st.write("Contenido adicional aquí.")