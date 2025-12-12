"""
Streamlit App for Speech Emotion Recognition and Speaker Identification
Supports real-time emotion detection and file upload for comprehensive analysis
"""

import streamlit as st
import numpy as np
import torch
import librosa
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
from datetime import datetime
import soundfile as sf
import sounddevice as sd
from io import BytesIO
import tempfile
import sys
import traceback

# Add project directory to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# Import preprocessing and models
from preprocessing import (
    InferencePreprocessor, InferenceEngine, ModelLoader
)
from models import AdvancedEmotionModel, AdvancedSpeakerModel

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Speech Emotion & Speaker Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .emotion-label {
        font-size: 1.5em;
        font-weight: bold;
        color: #1f77b4;
    }
    .confidence {
        font-size: 1.2em;
        color: #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING SETUP
# ============================================================================

@st.cache_resource
def load_models_internal():
    """Load models and preprocessing components (Cached)"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Use absolute paths
        project_dir = Path(__file__).parent.absolute()
        emotion_model_path = project_dir / 'Emotion_Recognition_best.pth'
        speaker_model_path = project_dir / 'Speaker_Identification_best.pth'
        emotion_encoder_path = project_dir / 'label_encoder_emotion.pkl'
        speaker_encoder_path = project_dir / 'label_encoder_speaker.pkl'
        
        print(f"\n{'='*60}")
        print("MODEL LOADING DETAILS")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Project directory: {project_dir}")
        print(f"\nFile sizes:")
        print(f"  Emotion model: {emotion_model_path.stat().st_size / (1024**2):.2f} MB")
        print(f"  Speaker model: {speaker_model_path.stat().st_size / (1024**2):.2f} MB")
        print(f"  Emotion encoder: {emotion_encoder_path.stat().st_size / 1024:.2f} KB")
        print(f"  Speaker encoder: {speaker_encoder_path.stat().st_size / 1024:.2f} KB")
        
        # Initialize components
        preprocessor = InferencePreprocessor(sr=22050, duration=3.0, device=device)
        model_loader = ModelLoader(device=device)
        
        # Model configuration (adjust based on your training)
        input_shape = (168, 104)
        num_emotions = 8
        num_speakers = 122
        
        print(f"\nModel configuration:")
        print(f"  Input shape: {input_shape}")
        print(f"  Num emotions: {num_emotions}")
        print(f"  Num speakers: {num_speakers}")
        
        print(f"\n{'='*60}")
        print("LOADING MODELS")
        print(f"{'='*60}")
        
        # Load models
        success = model_loader.load_models(
            emotion_model_path=str(emotion_model_path),
            speaker_model_path=str(speaker_model_path),
            emotion_encoder_path=str(emotion_encoder_path),
            speaker_encoder_path=str(speaker_encoder_path),
            emotion_model_class=AdvancedEmotionModel,
            speaker_model_class=AdvancedSpeakerModel,
            input_shape=input_shape,
            num_emotions=num_emotions,
            num_speakers=num_speakers
        )
        
        if not success:
            error_msg = model_loader.error_message or "Unknown error during model loading"
            print(f"\n❌ FAILURE: {error_msg}")
            print(f"{'='*60}\n")
            return None, None, error_msg
        
        # Set target feature shape
        preprocessor.set_target_feature_shape(input_shape)
        
        # Initialize inference engine
        inference_engine = InferenceEngine(
            emotion_model=model_loader.emotion_model,
            speaker_model=model_loader.speaker_model,
            label_encoder_emotion=model_loader.label_encoder_emotion,
            label_encoder_speaker=model_loader.label_encoder_speaker,
            device=device
        )
        
        print(f"\n✓ All models loaded successfully!")
        print(f"{'='*60}\n")
        return preprocessor, inference_engine, device
        
    except Exception as e:
        import traceback
        error_msg = f"Exception during model loading:\n{traceback.format_exc()}"
        print(f"\n❌ {error_msg}")
        print(f"{'='*60}\n")
        return None, None, error_msg

def load_models():
    """Wrapper to check files before loading"""
    project_dir = Path(__file__).parent.absolute()
    
    expected_files = {
        'Emotion_Recognition_best.pth': project_dir / 'Emotion_Recognition_best.pth',
        'Speaker_Identification_best.pth': project_dir / 'Speaker_Identification_best.pth',
        'label_encoder_emotion.pkl': project_dir / 'label_encoder_emotion.pkl',
        'label_encoder_speaker.pkl': project_dir / 'label_encoder_speaker.pkl'
    }
    
    missing_files = []
    for name, path in expected_files.items():
        if not path.exists():
            missing_files.append(name)
            
    if missing_files:
        st.error(f"❌ Missing files: {', '.join(missing_files)}")
        st.info("Please ensure these files are in the project directory:")
        st.code(str(project_dir))
        return None, None, None
        
    # If files exist, verify loading
    preprocessor, inference_engine, device_or_error = load_models_internal()
    
    if isinstance(device_or_error, str) and device_or_error not in ['cpu', 'cuda']:
        # It's an error message
        st.error("❌ Error Loading Models")
        st.error(device_or_error)
        with st.expander("📋 Check the terminal/console output for detailed debugging information"):
            st.info("Full error details have been printed to the console. Please check your terminal for the complete traceback.")
        return None, None, None
        
    return preprocessor, inference_engine, device_or_error


@st.cache_data
def get_emotion_emoji(emotion):
    """Get emoji for emotion"""
    emoji_map = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'neutral': '😐',
        'calm': '😌',
        'fear': '😨',
        'disgust': '😒',
        'surprise': '😲'
    }
    return emoji_map.get(emotion.lower(), '🎤')


def plot_emotion_distribution(emotion_probs, emotion_classes):
    """Plot emotion probability distribution"""
    emotions = emotion_classes
    probabilities = emotion_probs[0] * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probabilities,
            marker=dict(
                color=probabilities,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Emotion Classification Probabilities',
        xaxis_title='Emotion',
        yaxis_title='Probability (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_speaker_distribution(speaker_probs, speaker_classes, top_k=10):
    """Plot top-k speaker probabilities"""
    probs = speaker_probs[0] * 100
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    top_speakers = [speaker_classes[i] for i in top_indices]
    top_probs = probs[top_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_speakers,
            x=top_probs,
            orientation='h',
            marker=dict(
                color=top_probs,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{p:.1f}%' for p in top_probs],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_k} Speaker Candidates',
        xaxis_title='Probability (%)',
        yaxis_title='Speaker',
        height=500,
        hovermode='y unified'
    )
    
    return fig


def process_audio_file(audio_file, preprocessor):
    """Process uploaded audio file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Process audio
        features = preprocessor.process_audio_file(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        if features is None:
            st.error("Failed to process audio file")
            return None
        
        return features
        
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None


def record_and_process(preprocessor):
    """Record audio in real-time and process"""
    try:
        import sounddevice as sd
        
        st.info("Recording will start when you click the button below...")
        
        duration = st.slider("Recording Duration (seconds)", min_value=3, max_value=30, value=5, step=1)
        
        if st.button(f"🎙️ Start Recording ({duration} seconds)", key="record_btn"):
            sr = 22050
            
            with st.spinner(f"Recording for {duration} seconds..."):
                audio_data = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
                sd.wait()
            
            st.success("Recording completed!")
            
            # Save to temporary file
            # Windows fix: Close the file so sf.write can open it by name
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_path = tmp_file.name
                
                # Ensure audio is in correct format and range for WAV file
                # Flatten if needed (remove extra dimensions)
                audio_data_clean = np.squeeze(audio_data)
                
                # Ensure float32 is in [-1, 1] range
                max_val = np.max(np.abs(audio_data_clean))
                if max_val > 0:
                    audio_data_clean = audio_data_clean / max_val
                
                # Write as PCM 16-bit
                sf.write(tmp_path, audio_data_clean, sr, subtype='PCM_16')
                
                # Process audio
                features = preprocessor.process_audio_file(tmp_path)
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                features = None
            finally:
                # Clean up
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # Display the recorded audio for playback/verification
            st.subheader("🔊 Recorded Audio")
            st.audio(audio_data_clean, sample_rate=sr)
            
            if features is None:
                st.error("Failed to process recorded audio")
                return None
            
            return features
        
        return None
        
    except ImportError:
        st.warning("sounddevice package not installed. Please use file upload instead.")
        st.info("To enable recording, install: `pip install sounddevice soundfile`")
        return None
    except Exception as e:
        st.error(f"Error during recording: {e}")
        return None


# ============================================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================================

with st.sidebar:
    st.title("ℹ️ System Information")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.success(f"Device: {device.upper()}")
    
    # Model configuration
    st.subheader("Model Configuration")
    st.write("- **Emotion Classes**: 8")
    st.write("- **Speaker Classes**: 122")
    st.write("- **Sample Rate**: 22050 Hz")
    st.write("- **Audio Duration**: Variable (Segments of 3s)")
    
    st.divider()
    
    # About
    st.subheader("About")
    st.write("""
    This application uses advanced deep learning models to:
    - 🎭 Detect emotions from speech
    - 🎤 Identify speakers
    
    **Supported Emotions:**
    - Happy, Sad, Angry, Neutral
    - Calm, Fear, Disgust, Surprise
    """)


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<div class="main-header">🎤 Speech Emotion & Speaker Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Emotion Detection & Speaker Identification</div>', unsafe_allow_html=True)

# Load models
with st.spinner("Loading models... (this may take a moment)"):
    preprocessor, inference_engine, device = load_models()

if preprocessor is None or inference_engine is None:
    st.error("❌ Failed to load models. Please check if all required files are present:")
    st.error("""
    - Emotion_Recognition_best.pth
    - Speaker_Identification_best.pth
    - label_encoder_emotion.pkl
    - label_encoder_speaker.pkl
    """)
    st.stop()

# Mode selection
st.divider()
mode = st.radio(
    "Select Mode:",
    ["🎭 Emotion Detection (Real-time/Upload)", "🎤 Emotion + Speaker Analysis"],
    horizontal=True,
    label_visibility="collapsed"
)

# ============================================================================
# MODE 1: EMOTION DETECTION ONLY (Real-time or Upload)
# ============================================================================

if mode == "🎭 Emotion Detection (Real-time/Upload)":
    st.subheader("🎭 Emotion Detection")
    st.write("Detect emotions from speech without speaker identification.")
    
    st.divider()
    
    # Tab selection
    tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Record Audio"])
    
    with tab1:
        st.write("Upload an audio file (.wav, .mp3, .ogg) for emotion analysis")
        
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "flac"],
            key="emotion_upload"
        )
        
        if audio_file is not None:
            st.audio(audio_file)
            
            if st.button("🔍 Analyze Emotion", key="emotion_analyze"):
                with st.spinner("Processing audio..."):
                    features = process_audio_file(audio_file, preprocessor)
                
                if features is not None:
                    with st.spinner("Running emotion detection..."):
                        emotion, confidence, emotion_probs = inference_engine.predict_emotion(features)
                    
                    if emotion is not None:
                        # Display results
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            st.markdown("")
                            st.markdown("")
                            emoji = get_emotion_emoji(emotion)
                            st.markdown(f"<div style='text-align: center; font-size: 3em;'>{emoji}</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="emotion-box">', unsafe_allow_html=True)
                            st.markdown(f'<div class="emotion-label">Detected Emotion</div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="font-size: 2em; color: #1f77b4; margin: 10px 0;">{emotion.upper()}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("")
                            st.markdown("")
                        
                        # Plot probabilities
                        st.divider()
                        emotion_classes = inference_engine.label_encoder_emotion.classes_
                        fig = plot_emotion_distribution(emotion_probs, emotion_classes)
                        st.plotly_chart(fig, use_container_width=True, key="emotion_upload_chart")
                        
                        # Download results
                        st.divider()
                        result_text = f"""
Emotion Detection Results
=========================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Detected Emotion: {emotion}
Confidence: {confidence:.2f}%

Emotion Probabilities:
"""
                        for em, prob in zip(emotion_classes, emotion_probs[0]):
                            result_text += f"{em.capitalize()}: {prob*100:.2f}%\n"
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=result_text,
                            file_name=f"emotion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
    
    with tab2:
        st.write("Record audio in real-time for emotion analysis")
        
        # Initialize session state for recording
        if "recorded_features" not in st.session_state:
            st.session_state.recorded_features = None
        if "recorded_audio" not in st.session_state:
            st.session_state.recorded_audio = None
            st.session_state.recorded_sr = None
        
        # Recording UI
        duration = st.slider("Recording Duration (seconds)", min_value=3, max_value=30, value=5, step=1, key="emotion_duration")
        
        if st.button(f"🎙️ Start Recording ({duration} seconds)", key="emotion_record_btn"):
            sr = 22050
            
            with st.spinner(f"Recording for {duration} seconds..."):
                audio_data = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
                sd.wait()
            
            st.success("Recording completed!")
            
            # Save to temporary file
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_path = tmp_file.name
                
                # Ensure audio is in correct format and range for WAV file
                audio_data_clean = np.squeeze(audio_data)
                
                # Ensure float32 is in [-1, 1] range
                max_val = np.max(np.abs(audio_data_clean))
                if max_val > 0:
                    audio_data_clean = audio_data_clean / max_val
                
                # Write as PCM 16-bit
                sf.write(tmp_path, audio_data_clean, sr, subtype='PCM_16')
                
                # Process audio
                features = preprocessor.process_audio_file(tmp_path)
                
                # Store in session state
                st.session_state.recorded_features = features
                st.session_state.recorded_audio = audio_data_clean
                st.session_state.recorded_sr = sr
                
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                st.session_state.recorded_features = None
            finally:
                # Clean up
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        # Display recorded audio if available
        if st.session_state.recorded_audio is not None:
            st.subheader("🔊 Recorded Audio")
            st.audio(st.session_state.recorded_audio, sample_rate=st.session_state.recorded_sr)
            
            # Show analyze button if features were successfully processed
            if st.session_state.recorded_features is not None:
                if st.button("🔍 Analyze Recorded Audio", key="emotion_record_analyze"):
                    with st.spinner("Running emotion detection..."):
                        emotion, confidence, emotion_probs = inference_engine.predict_emotion(st.session_state.recorded_features)
                    
                    if emotion is not None:
                        # Display results
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            st.markdown("")
                            st.markdown("")
                            emoji = get_emotion_emoji(emotion)
                            st.markdown(f"<div style='text-align: center; font-size: 3em;'>{emoji}</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="emotion-box">', unsafe_allow_html=True)
                            st.markdown(f'<div class="emotion-label">Detected Emotion</div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="font-size: 2em; color: #1f77b4; margin: 10px 0;">{emotion.upper()}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("")
                            st.markdown("")
                        
                        # Plot probabilities
                        st.divider()
                        emotion_classes = inference_engine.label_encoder_emotion.classes_
                        fig = plot_emotion_distribution(emotion_probs, emotion_classes)
                        st.plotly_chart(fig, use_container_width=True, key="emotion_record_chart")

# ============================================================================
# MODE 2: EMOTION + SPEAKER ANALYSIS
# ============================================================================

else:  # mode == "🎤 Emotion + Speaker Analysis"
    st.subheader("🎤 Comprehensive Analysis")
    st.write("Analyze both emotion and speaker from audio files.")
    
    st.divider()
    
    audio_file = st.file_uploader(
        "Choose an audio file for comprehensive analysis",
        type=["wav", "mp3", "ogg", "flac"],
        key="comprehensive_upload"
    )
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("🔍 Analyze Audio", key="comprehensive_analyze"):
            with st.spinner("Processing audio..."):
                features = process_audio_file(audio_file, preprocessor)
            
            if features is not None:
                with st.spinner("Running analysis..."):
                    results = inference_engine.predict_both(features)
                
                # Display results
                st.divider()
                st.subheader("📊 Analysis Results")
                
                # Emotion Results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎭 Emotion Detection")
                    emotion = results['emotion']
                    emotion_conf = results['emotion_confidence']
                    emoji = get_emotion_emoji(emotion)
                    
                    st.markdown(f"<div style='text-align: center; font-size: 2.5em;'>{emoji}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 1.5em; font-weight: bold; color: #1f77b4;'>{emotion.upper()}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 1.2em; color: #2ecc71;'>Confidence: {emotion_conf:.2f}%</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 🎤 Speaker Identification")
                    speaker = results['speaker']
                    speaker_conf = results['speaker_confidence']
                    
                    st.markdown(f"<div style='text-align: center; font-size: 1.5em; font-weight: bold; color: #1f77b4;'>{speaker}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 1.2em; color: #2ecc71;'>Confidence: {speaker_conf:.2f}%</div>", unsafe_allow_html=True)
                
                # Detailed charts
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Emotion Probabilities")
                    emotion_classes = inference_engine.label_encoder_emotion.classes_
                    fig_emotion = plot_emotion_distribution(results['emotion_probabilities'], emotion_classes)
                    st.plotly_chart(fig_emotion, use_container_width=True, key="emotion_comprehensive_chart")
                
                with col2:
                    st.subheader("Speaker Probabilities")
                    speaker_classes = inference_engine.label_encoder_speaker.classes_
                    fig_speaker = plot_speaker_distribution(results['speaker_probabilities'], speaker_classes, top_k=15)
                    st.plotly_chart(fig_speaker, use_container_width=True, key="speaker_comprehensive_chart")
                
                # Download results
                st.divider()
                result_text = f"""
Comprehensive Analysis Results
==============================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EMOTION DETECTION
-----------------
Detected Emotion: {results['emotion']}
Confidence: {results['emotion_confidence']:.2f}%

Emotion Probabilities:
"""
                for em, prob in zip(emotion_classes, results['emotion_probabilities'][0]):
                    result_text += f"  {em.capitalize()}: {prob*100:.2f}%\n"
                
                result_text += f"""
SPEAKER IDENTIFICATION
----------------------
Identified Speaker: {results['speaker']}
Confidence: {results['speaker_confidence']:.2f}%

Top 10 Speaker Candidates:
"""
                speaker_probs = results['speaker_probabilities'][0] * 100
                top_indices = np.argsort(speaker_probs)[-10:][::-1]
                for i, idx in enumerate(top_indices, 1):
                    result_text += f"  {i}. {speaker_classes[idx]}: {speaker_probs[idx]:.2f}%\n"
                
                st.download_button(
                    label="📥 Download Full Report",
                    data=result_text,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
<div style='text-align: center; color: #999; font-size: 0.9em;'>
Built with ❤️ using Streamlit, PyTorch, and Deep Learning
</div>
""", unsafe_allow_html=True)
