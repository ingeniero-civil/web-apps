import streamlit as st

def run():
    st.set_page_config(
        page_title="Home",
        page_icon="👷‍♂️",
    )

    st.write("# Bienvenidos! 👷‍♂️")

    st.sidebar.success("Selecciona una de las opciones 👆🏽")

    st.markdown(
        """
        ##### En esta página encontrarán diversas aplicaciones desarrolladas por mí que espero les sean útiles.
        Para cualquier comentario o sugerencia, pueden contactarme en mis redes:
        - [Twitter] (https://twitter.com/JavierCornejoT)
        - [Instagram] (https://www.instagram.com/javier.cornejot/)
        - [LinkedIn] (https://www.linkedin.com/in/jcornejot/)
    """
    )


if __name__ == "__main__":
    run()
