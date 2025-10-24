import streamlit as st
import os
import google.generativeai as genai
from youtube_analyzer_crew import create_crew
import tempfile
import uuid
import yt_dlp
import shutil  # <-- Import the shell utilities library
import time     # <-- Import time for the waiting loop

# --- Gemini API Configuration ---
# Load API key from Streamlit secrets or .env file
try:
    # This is for Streamlit Cloud
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
except KeyError:
    # This is for local development
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
        st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- UPDATED: Download function using yt-dlp ---
def download_audio_from_youtube(youtube_url, temp_dir):
    """
    Downloads audio from a YouTube URL using yt-dlp and saves it as an mp3.
    Returns the file path to the downloaded audio file.
    """
    # Generate a unique file name *base*
    file_name_base = f"{uuid.uuid4()}"
    temp_audio_path_base = os.path.join(temp_dir, file_name_base)
    final_mp3_path = f"{temp_audio_path_base}.mp3"

    # yt-dlp options
    # We need ffmpeg installed for this to work
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl': temp_audio_path_base, # Use the base path, yt-dlp adds .mp3
        'noplaylist': True,
    }

    try:
        st.write(f"Attempting to download audio from: {youtube_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Check if the final mp3 file was created
        if not os.path.exists(final_mp3_path):
            st.error(f"yt-dlp processing error. Expected file not found: {final_mp3_path}")
            st.warning("This could be a download or conversion issue.")
            return None

        st.write(f"Audio downloaded and converted to: {final_mp3_path}")
        return final_mp3_path # Return the full .mp3 path
    except Exception as e:
        st.error(f"Error downloading YouTube video with yt-dlp: {e}")
        st.warning("This can happen if the video is private, deleted, or geographically restricted. It also requires `ffmpeg` to be installed on the system.")
        return None

# --- UPDATED: Gemini Transcription Function ---
def transcribe_video_with_gemini(video_file_path, language_code):
    """
    Transcribes the video file using Gemini API.
    """
    st.write(f"Transcribing {video_file_path}...")
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    # Upload the file to the Gemini API
    # Note: Gemini has a file size limit, but it's generally high for audio.
    try:
        st.write("Uploading file to Gemini... This may take a moment.")
        audio_file = genai.upload_file(path=video_file_path)
        
        # --- ADDED: Wait for Gemini to process the file ---
        while audio_file.state.name == "PROCESSING":
            st.write("Waiting for Gemini file processing...")
            time.sleep(2) # Wait 2 seconds and check again
        
        if audio_file.state.name == "FAILED":
            st.error(f"Gemini file processing failed. State: {audio_file.state.name}")
            return None, None
        # --- END ADDED BLOCK ---

        st.write("Audio file uploaded and processed by Gemini.")
    except Exception as e:
        st.error(f"Error uploading file to Gemini: {e}")
        st.error("This might be due to an unsupported audio format or a file size limit.")
        return None, None

    # Create the prompt for transcription
    prompt = f"Please transcribe the following audio. The audio is in {language_code}. Provide only the full, clean transcription and nothing else."
    
    try:
        response = model.generate_content([prompt, audio_file])
        st.write("Transcription received.")
        return response.text, audio_file
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None, None

# --- Main Application ---
def main():
    st.set_page_config(page_title="YouTube Compliance Analyzer", layout="wide")
    st.title("🤖 YouTube Crypto Compliance AI Analyzer")

    st.sidebar.header("About")
    st.sidebar.info(
        "This tool uses a CrewAI team of AI agents, powered by Google Gemini, "
        "to analyze your YouTube video's transcription against YouTube's Community Guidelines, "
        "specifically for cryptocurrency content."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Provide Your Video")
        
        youtube_url = st.text_input("Enter YouTube Video URL")
        
        st.markdown("<p style='text-align: center; color: grey;'>OR</p>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload a video or audio file", type=["mp4", "mp3", "wav", "m4a", "flac"])
        
        language = st.selectbox(
            "Select Video Language",
            ("English", "Spanish", "German", "Portuguese", "French", "Arabic")
        )

        analyze_button = st.button("Analyze Video", type="primary")

    with col2:
        st.subheader("2. AI Analysis & Report")
        
        if analyze_button:
            if not youtube_url and not uploaded_file:
                st.error("Please provide a YouTube URL or upload a file.")
                st.stop()
            
            # Create a temporary directory to store files
            temp_dir = tempfile.mkdtemp()
            file_path = None
            
            try:
                if youtube_url:
                    # Download from YouTube URL
                    with st.spinner("Downloading audio from YouTube... (This may take a moment)"):
                        file_path = download_audio_from_youtube(youtube_url, temp_dir)
                
                elif uploaded_file:
                    # Save uploaded file
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"File uploaded: {uploaded_file.name}")
                
                if file_path and os.path.exists(file_path):
                    # --- Step 1: Transcribe ---
                    with st.spinner(f"Transcribing {language} audio... (This can take a while for large files)"):
                        transcription, audio_file_reference = transcribe_video_with_gemini(file_path, language)
                    
                    if transcription:
                        st.success("Transcription complete.")
                        
                        # --- Step 2: Analyze (FIXED) ---
                        with st.spinner("AI Crew is analyzing the content..."):
                            # Create the crew, passing the transcription and language as arguments
                            analyzer_crew = create_crew(
                                video_transcript=transcription,
                                video_language=language
                            )
                            
                            # Kick off the analysis task
                            # The inputs are now passed at creation, so kickoff() needs no inputs.
                            report = analyzer_crew.kickoff()
                        
                        st.subheader("Compliance Report")
                        st.markdown(report)
                        
                        with st.expander("Show Full Transcription"):
                            st.text_area("", transcription, height=300)
                    
                    else:
                        st.error("Could not generate transcription. Analysis cancelled.")

            finally:
                # --- Step 3: Cleanup (Updated) ---
                # Use shutil.rmtree to forcefully delete the directory and all its contents
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
                # We can't delete the Gemini file reference here as it's not a local file

if __name__ == "__main__":
    main()

