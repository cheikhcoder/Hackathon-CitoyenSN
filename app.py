
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Charger les variables d'environnement
load_dotenv()

# Charger le modèle et le tokenizer de traduction Wolof ↔ Français
model_name = "cifope/nllb-200-wo-fr-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Initialiser OpenAI pour la transcription audio
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fonction de transcription audio
def transcribe_audio(audio_file):
    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )
    return transcription['text']

# Fonction de traduction du français vers le wolof
def translate_to_wolof(text):
    tokenizer.src_lang = "fra_Latn"
    tokenizer.tgt_lang = "wol_Latn"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("wol_Latn"),
        max_new_tokens=1000
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

# Initialisation du chatbot
def get_chatbot(language, vectorstore):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Fonction pour extraire le texte des fichiers PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Fonction pour diviser le texte en morceaux
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        separator="\n",
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Fonction pour créer le vecteurstore
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Résumer les documents PDF
def summarize_text(text):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"Résumé du texte suivant : {text}"
    response = llm.predict(prompt)
    return response

# Page d'accueil
def homepage():
    st.title("Bienvenue sur CitoyenSN")
    st.image("logo.jpg", width=150)  # Logo en haut
    st.markdown("""
    ### Projet PatrioteSN
    PatrioteSN est un assistant juridique interactif qui offre des informations sur le droit sénégalais en utilisant l'IA.
    Vous pouvez poser des questions au chatbot, en français, anglais, russe ou wolof, et obtenir des réponses rapides et précises.

    Vous pouvez aussi télécharger des documents juridiques pour obtenir un résumé en quelques secondes.
    """)

# Page du Chatbot avec support audio et option de traduction en Wolof
def chatbot_page():
    st.title("CitoyenSN - Chatbot Juridique avec Transcription Audio et Traduction Wolof")

    # Sélection de la langue pour la conversation
    language = st.selectbox("Choisissez la langue", ["Français", "Anglais", "Russe"])

    # Instructions basées sur la langue choisie
    language_prompts = {
        "Français": "Réponds en français : ",
        "Anglais": "Respond in English: ",
        "Russe": "Ответь на русском: "
    }
    prompt_prefix = language_prompts[language]

    # Champ de saisie pour la question texte
    user_question = st.text_input("Posez une question sur le droit sénégalais (ou téléversez un fichier audio)")

    # Téléversement de fichier audio
    audio_file = st.file_uploader("Ou téléversez un fichier audio", type=["mp3", "wav", "m4a"])

    # Traiter la question texte ou audio
    if "vectorstore" not in st.session_state:
        with st.spinner("Chargement des fichiers PDF..."):
            pdf_directory = "dossiers"
            pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
            if pdf_files:
                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = get_vectorstore(text_chunks)

    # Initialiser le chatbot
    st.session_state.conversation = get_chatbot(language, st.session_state.vectorstore)

    # Si l'utilisateur a entré une question texte
    if user_question:
        question = f"{prompt_prefix}{user_question}"
        response = st.session_state.conversation({'question': question})
        response_text = response['answer']
        st.markdown("### Réponse en Français :")
        st.write(response_text)

        # Option de traduction en Wolof
        if st.button("Traduire en Wolof"):
            wolof_translation = translate_to_wolof(response_text)
            st.markdown("### Réponse en Wolof :")
            st.write(wolof_translation)

    # Si un fichier audio est téléversé, transcrire et envoyer au chatbot
    elif audio_file is not None:
        with st.spinner("Transcription de l'audio..."):
            transcription = transcribe_audio(audio_file)
            question = f"{prompt_prefix}{transcription}"
            response = st.session_state.conversation({'question': question})
            response_text = response['answer']
            st.markdown("### Réponse en Français :")
            st.write(response_text)

            # Option de traduction en Wolof
            if st.button("Traduire en Wolof"):
                wolof_translation = translate_to_wolof(response_text)
                st.markdown("### Réponse en Wolof :")
                st.write(wolof_translation)

# Page de résumé de documents
def summarize_page():
    st.title("Résumé de Documents Juridiques")
    uploaded_files = st.file_uploader("Téléchargez vos fichiers PDF pour un résumé", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Résumé en cours..."):
            raw_text = get_pdf_text(uploaded_files)
            summary = summarize_text(raw_text)
            st.markdown("### Résumé :")
            st.write(summary)

# Interface principale pour la navigation entre pages
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Chatbot Juridique avec Audio", "Résumé de Documents"])

    if page == "Accueil":
        homepage()
    elif page == "Chatbot Juridique avec Audio":
        chatbot_page()
    elif page == "Résumé de Documents":
        summarize_page()

if __name__ == "__main__":
    main()
