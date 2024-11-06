import streamlit as st
from utils import get_pdf_text, summarize_text, translate_to_english_russian, translate_to_wolof

def summarize_page():
    st.title("Résumé de Documents Juridiques")

    # PDF file uploader
    uploaded_files = st.file_uploader("Téléchargez vos fichiers PDF pour un résumé", type="pdf", accept_multiple_files=True)

    # Summarize each uploaded file
    if uploaded_files:
        with st.spinner("Résumé en cours..."):
            raw_text = get_pdf_text(uploaded_files)
            summary = summarize_text(raw_text)
            st.markdown("### Résumé en Français :")
            st.write(summary)

            # Display translation buttons in columns
            col1, col2, col3 = st.columns(3)
            show_english = col1.button("Traduire en Anglais")
            show_russian = col2.button("Traduire en Russe")
            show_wolof = col3.button("Traduire en Wolof")

            # Display each translation below without columns
            if show_english:
                english_translation = translate_to_english_russian(summary, "English")
                st.markdown("### Résumé en Anglais :")
                st.write(english_translation)

            if show_russian:
                russian_translation = translate_to_english_russian(summary, "Russian")
                st.markdown("### Résumé en Russe :")
                st.write(russian_translation)

            if show_wolof:
                wolof_translation = translate_to_wolof(summary)
                st.markdown("### Résumé en Wolof :")
                st.write(wolof_translation)

# Call the function to display the page content when the file loads
summarize_page()
