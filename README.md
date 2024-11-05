
# PatrioteSN - Legal Assistant Chatbot

**PatrioteSN** is an interactive legal assistant leveraging artificial intelligence to provide information on Senegalese law. This project includes various functionalities: a multilingual chatbot supporting audio transcription, document summarization, and translation of responses into Wolof.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies and Models Used](#technologies-and-models-used)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Running the Project](#running-the-project)
6. [Usage](#usage)
7. [Acknowledgments](#acknowledgments)

---

## Project Overview

**PatrioteSN** is designed to assist users in getting quick answers to legal questions related to Senegalese law. The project provides:

- **Multilingual Chatbot**: Users can ask legal questions in French, English, Russian, or Wolof, and receive responses in the same language.
- **Audio Transcription**: Users can upload audio files with questions, which are transcribed to text.
- **Document Summarization**: Users can upload PDF documents to get concise summaries.
- **Translation to Wolof**: French responses can be translated into Wolof using a custom translation model.

## Technologies and Models Used

### Libraries

- **Streamlit**: For building the web application interface.
- **PyPDF2**: To extract text from PDF files for summarization.
- **LangChain**: Provides tools for chatbot functionality, embeddings, and vector storage.
- **Hugging Face Transformers**: For translation using the `cifope/nllb-200-wo-fr-distilled-600M` model.
- **OpenAI API**: Used for the Whisper model for audio transcription and GPT models for language generation.

### Models

1. **Whisper Model (OpenAI)**: Used for converting audio input to text.
2. **NLLB-200 Model (Hugging Face)**: `cifope/nllb-200-wo-fr-distilled-600M` for French-to-Wolof translation.
3. **OpenAI ChatGPT**: Provides conversational AI responses to questions in various languages.

### Techniques

1. **Text Splitting and Chunking**: Splits large text into smaller chunks for efficient processing.
2. **Vector Storage with FAISS**: Stores embeddings for retrieval.
3. **Conversational Memory**: Maintains conversation context using LangChain’s `ConversationBufferMemory`.

---

## Project Structure

```plaintext
PatrioteSN
│
├── main.py                    # Main Streamlit app file
├── requirements.txt           # Required Python packages
├── .env                       # Environment file for API keys
├── dossiers                   # Directory containing uploaded documents
│   ├── sample_document1.pdf   # Sample PDF files
│   └── sample_document2.pdf   # Additional documents
└── README.md                  # Project documentation
```

---

## Setup and Installation

### Prerequisites

- **Python 3.8+** installed on your system.
- Access to OpenAI API and Hugging Face API for model usage.

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/PatrioteSN.git
   cd PatrioteSN
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   - Create a `.env` file in the root directory and add your API keys:
     ```plaintext
     OPENAI_API_KEY=your_openai_key
     HUGGINGFACE_HUB_TOKEN=your_huggingface_token
     ```

5. **Download Models (Optional)**
   - The Hugging Face model for Wolof translation will be downloaded automatically the first time it’s used. However, you can download it manually if desired.

---

## Running the Project

To start the Streamlit application:

```bash
streamlit run main.py
```

The app will open in your default web browser, showing the homepage with navigation options.

---

## Usage

### 1. **Homepage**
   - Introduces the project and provides navigation options for the chatbot and document summarization.

### 2. **Chatbot Page**
   - **Select Language**: Choose a language for interaction (French, English, Russian, or Wolof).
   - **Ask a Question**: Type a question or upload an audio file (MP3, WAV, M4A). The app will transcribe the audio using Whisper and provide a response.
   - **Translate to Wolof**: After receiving a response in French, click "Translate to Wolof" to view the answer in Wolof.

### 3. **Document Summarization**
   - **Upload PDF**: Place PDF files in the `dossiers` folder to be summarized. 
   - **Summarize**: The app will provide a concise summary of each document uploaded in the `dossiers` folder.

---

## Code Explanation

1. **Audio Transcription**:
   - The `transcribe_audio()` function uses OpenAI's Whisper model to convert audio input to text.

2. **Chatbot**:
   - The chatbot uses OpenAI's ChatGPT model, with LangChain’s `ConversationalRetrievalChain` and `ConversationBufferMemory` for conversational context.

3. **Translation**:
   - The `translate_to_wolof()` function utilizes the Hugging Face `cifope/nllb-200-wo-fr-distilled-600M` model for French-to-Wolof translation.

4. **PDF Summarization**:
   - Text is extracted from PDF files in the `dossiers` folder using `PyPDF2`, split into chunks, and summarized using OpenAI’s language model.

---

## Acknowledgments

- **OpenAI** for providing Whisper and GPT models.
- **Hugging Face** for the NLLB-200 model for Wolof translation.
- **LangChain** for tools to handle conversation memory and vector storage.
- **Streamlit** for making the development of interactive web applications simple.

---

## Future Improvements

- **Enhanced Wolof Translation**: Improve translation quality by fine-tuning the model or adding more Wolof training data.
- **Language Support Expansion**: Add more languages relevant to Senegal.
- **Advanced Legal Document Analysis**: Extend summarization with entity extraction and deeper legal insights.

