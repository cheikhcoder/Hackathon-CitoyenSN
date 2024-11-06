import os
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import streamlit as st


# Resolve potential OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

# Load the Wolof ↔ French translation model and tokenizer
model_name = "cifope/nllb-200-wo-fr-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# French-to-Wolof translation function
def translate_to_wolof(text):
    tokenizer.src_lang = "fra_Latn"
    tokenizer.tgt_lang = "wol_Latn"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("wol_Latn"),
        max_new_tokens=2000
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

# English and Russian translation function using OpenAI
def translate_to_english_russian(text, target_language):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"Translate the following text to {target_language}: {text}"
    response = llm.predict(prompt)
    return response

# Initialize the chatbot with memory and retrieval capabilities
def get_chatbot(language, vectorstore):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# PDF text extraction function
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20, separator="\n", length_function=len)
    return text_splitter.split_text(text)

# Create a vector store from text chunks for retrieval
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Summarization function using OpenAI's GPT model
def summarize_text(text):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"Résumé du texte suivant : {text}"
    return llm.predict(prompt)

# Audio transcription function using Whisper

# Audio recording widget (JavaScript for recording audio in the browser)
