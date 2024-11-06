# chatbot_page.py

import streamlit as st
from utils import get_chatbot, translate_to_wolof, get_pdf_text, get_text_chunks, get_vectorstore
import os
def chatbot_page():
    st.title("CitoyenSN - Chatbot Juridique avec Traduction Wolof")
    
    # Language selection
    language = st.selectbox("Choisissez la langue", ["Français", "Anglais", "Russe"])
    language_prompts = {
        "Français": "Réponds en français : ",
        "Anglais": "Respond in English: ",
        "Russe": "Ответь на русском: "
    }
    prompt_prefix = language_prompts[language]
    
    # User question input
    user_question = st.text_input("Posez une question sur le droit sénégalais")

    # Initialize vectorstore if not loaded
    if "vectorstore" not in st.session_state:
        with st.spinner("Chargement des fichiers PDF..."):
            pdf_directory = "dossiers"
            pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
            if pdf_files:
                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = get_vectorstore(text_chunks)

    # Initialize the chatbot
    st.session_state.conversation = get_chatbot(language, st.session_state.vectorstore)

    # Process user question
    if user_question:
        question = f"{prompt_prefix}{user_question}"
        response = st.session_state.conversation({'question': question})
        response_text = response['answer']
        st.markdown("### Réponse ")
        st.write(response_text)

        # Translate to Wolof option
        if st.button("Traduire en Wolof"):
            wolof_translation = translate_to_wolof(response_text)
            st.markdown("### Réponse en Wolof :")
            st.write(wolof_translation)

# Call the function to display the page content when the file loads
chatbot_page()
