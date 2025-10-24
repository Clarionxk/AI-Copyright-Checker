import streamlit as st
import os
import tempfile
import google.generativeai as genai
from youtube_analyzer_crew import create_crew
from pytube import YouTube
import time

# --- 1. Transcription Function (Using Gemini API) ---

def transcribe_video_with_gemini(video_path, language_name):
    """
    Transcribes the given video file (at a local path) using the Google Gemini API.

    Args:
        video_path (str): The local file path to the video/audio file.
        language_name (str): The full language name (e.g., 'Spanish', 'German').

    Returns:
        str: The full transcript of the video, or None if failed.
    """
    st.info(f"Transcribing video in {language_name}... This may take a few minutes.")
    
    # Configure the Gemini client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY environment variable not set.")
        return None
        
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model="gemini-2.5-flash-preview-09-2025")

    uploaded_file = None
    try:
        # 1. Upload the file to the Gemini Files API
        st.info("Uploading file to Gemini API...")
        uploaded_file = genai.upload_file(path=video_path)
        
        # 2. Make the transcription request
        st.info("File uploaded. Requesting transcription...")
        prompt = f"Generate a full, verbatim transcript of this video. The speech is in {language_name}."
        
        response = client.generate_content([prompt, uploaded_file])
        
        # Clean up the uploaded file from Gemini's servers
        genai.delete_file(uploaded_file.name)
        
        return response.text

    except Exception as e:
        st.error(f"Error during Gemini transcription: {e}")
        st.error("Please ensure your GOOGLE_API_KEY is correct and has access to the Gemini API.")
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception as del_e:
                st.warning(f"Could not clean up remote file: {del_e}")
        return None

# --- 2. YouTube Download Function ---

def download_audio_from_youtube(url, temp_dir):
    """
    Downloads the audio from a YouTube URL to a temporary file.

    Args:
        url (str): The YouTube video URL.
        temp_dir (str): The temporary directory to save the file.

    Returns:
        str: The file path to the downloaded audio file, or None if failed.
    """
    try:
        st.info("Downloading audio from YouTube...")
        yt = YouTube(url)
        
        # Filter for audio-only streams and select the first one
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if not audio_stream:
            st.error("Could not find an audio-only stream for this video.")
            return None
        
        # Download the audio to the temporary directory
        output_path = audio_stream.download(output_path=temp_dir)
        st.success(f"Successfully downloaded audio: {yt.title}")
        return output_path
        
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None

# --- 3. Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("🤖 YouTube Crypto Compliance Analyzer (Powered by Gemini)")
st.markdown("""
Analyze your video for YouTube Community Guideline violations.
You can either paste a YouTube URL or upload a video file directly.
""")

# Language selection
language_options = {
    "Spanish": "es",
    "German": "de",
    "Portuguese": "pt",
    "French": "fr",
    "Arabic": "ar",
    "English": "en"
}
selected_language_name = st.selectbox(
    "Select the language of the video:",
    options=language_options.keys()
)

st.divider()

# --- Input Method: YouTube URL ---
youtube_url = st.text_input("Option 1: Paste a YouTube URL")

st.markdown("<p style='text-align: center; color: grey;'>OR</p>", unsafe_allow_html=True)

# --- Input Method: File Upload ---
uploaded_file = st.file_uploader("Option 2: Upload your video file (.mp4, .mov, .webm, .mp3, .wav)", type=["mp4", "mov", "webm", "mp3", "wav", "m4a", "flac"])

st.divider()

# Analysis button
if st.button("Analyze Video"):
    if not youtube_url and not uploaded_file:
        st.warning("Please either paste a YouTube URL or upload a file.")
    elif not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("SERPER_API_KEY"):
        st.error("API Key not found. Please set both GOOGLE_API_KEY and SERPER_API_KEY environment variables.")
    else:
        with st.spinner("Analyzing... This involves downloading (if URL), transcription, and AI crew analysis. Please wait."):
            
            temp_dir = tempfile.mkdtemp()
            video_path_to_analyze = None
            cleanup_path = None

            try:
                # --- Step 1: Get the video file path ---
                if youtube_url:
                    # Download from YouTube
                    video_path_to_analyze = download_audio_from_youtube(youtube_url, temp_dir)
                    cleanup_path = video_path_to_analyze # Mark for cleanup
                
                elif uploaded_file:
                    # Save uploaded file to a temporary path
                    video_path_to_analyze = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path_to_analyze, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    cleanup_path = video_path_to_analyze # Mark for cleanup

                # --- Step 2: Transcribe the video ---
                if video_path_to_analyze:
                    transcript = transcribe_video_with_gemini(video_path_to_analyze, selected_language_name)
                    
                    if transcript:
                        st.subheader(f"Generated Transcript ({selected_language_name})")
                        st.text_area("", transcript, height=150)
                        
                        # --- Step 3: Run the CrewAI analysis ---
                        st.subheader("🤖 AI Compliance Report")
                        with st.spinner("AI crew is analyzing the transcript..."):
                            try:
                                crew = create_crew(transcript, selected_language_name)
                                result = crew.kickoff()
                                st.markdown(result)
                            except Exception as e:
                                st.error(f"An error occurred during AI analysis: {e}")
                                st.error("This often happens due to API key issues or rate limits.")
            
            finally:
                # --- Step 4: Cleanup ---
                if cleanup_path and os.path.exists(cleanup_path):
                    os.remove(cleanup_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)

