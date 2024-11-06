import streamlit as st

# Set up the main page configuration
st.set_page_config(page_title="CitoyenSN", page_icon="🌍")

def main():
    # Homepage content
    st.title("🌍 Bienvenue sur CitoyenSN")
    st.image("logo.jpg", width=150)
    st.markdown("## Votre Assistant Juridique Interactif")
    st.markdown("""
    CitoyenSN est une plateforme innovante qui utilise l'IA pour rendre le droit accessible à tous les citoyens sénégalais. 
    Posez vos questions en **français**, **anglais**, **russe**, ou **wolof** et recevez des réponses précises.
    """)

    st.markdown("### 🚀 Fonctionnalités Principales")
    st.markdown("""
    - 🗣 **Chatbot Multilingue** : Obtenez des réponses à vos questions juridiques.
    - 📄 **Résumé de Documents** : Téléchargez des documents pour un résumé concis.
    - 🌐 **Traduction en Wolof** : Accédez aux informations en Wolof.
    - 🎙 **Transcription Audio** : Posez vos questions oralement.
    """)

    st.markdown("### 🤔 Pourquoi CitoyenSN ?")
    st.markdown("""
    CitoyenSN vise à démocratiser l'accès à l'information juridique au Sénégal. Que vous soyez un professionnel du droit ou un citoyen 
    cherchant à comprendre vos droits, CitoyenSN vous simplifie l'accès aux informations juridiques essentielles.
    """)

    st.markdown("---")
    st.markdown("**Commencez dès maintenant** en explorant le menu à gauche pour poser vos questions ou obtenir un résumé de vos documents !")

if __name__ == "__main__":
    main()



