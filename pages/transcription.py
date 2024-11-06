import streamlit as st
import time  # For simulating delay

def main():
    st.title("🎤  Audio Transcription")

    # Step 1: Upload Audio File
    st.subheader("Step 1: Upload an Audio File")
    audio_file = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

    if audio_file is not None:
        st.success("Audio file uploaded successfully!")

    
        st.subheader("Step 2:  Transcription")
        if st.button("Transcribe"):
            with st.spinner(" transcription..."):
                time.sleep(10)  # Simulate a 30-second processing delay
            st.success("Transcription completed!")
            mock_transcription = "CitoyenSN is designed as an accessible, AI-powered legal assistant that addresses the critical need for clear, understandable legal information for Senegalese citizens."
            st.markdown("### Transcription Result:")
            st.write(mock_transcription)

if __name__ == "__main__":
    main()

# import streamlit as st
# import openai
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def transcribe_audio(file):
#     """
#     Transcribes the uploaded audio file using OpenAI's Whisper model.
    
#     Args:
#         file: Uploaded audio file.
        
#     Returns:
#         str: Transcribed text.
#     """
#     try:
#         response = openai.Audio.transcribe(
#             model="whisper-1",
#             file=file
#         )
#         return response['text']
#     except Exception as e:
#         return f"An error occurred: {e}"

# # Streamlit app layout
# st.title("Audio Transcription App")
# st.markdown("Upload an audio file and get the transcription using OpenAI's Whisper model.")

# # File uploader
# uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])

# if uploaded_file:
#     st.audio(uploaded_file, format="audio/mp3")
#     st.markdown("### Transcription in Progress...")
    
#     # Transcribe audio file
#     with st.spinner("Transcribing..."):
#         transcription = transcribe_audio(uploaded_file)
    
#     st.markdown("### Transcription Result:")
#     st.write(transcription)
