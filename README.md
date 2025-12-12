# 🎤 Speech Emotion & Speaker Recognition System

A state-of-the-art Deep Learning application for real-time Speech Emotion Recognition (SER) and Speaker Identification (SI). Built with **PyTorch**, **Streamlit**, and **Librosa**, this system leverages advanced CNN-LSTM-Attention architectures to deliver high-accuracy analysis of audio signals.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## ✨ Features

-   **🎭 Real-time Emotion Detection**: Record audio directly from your microphone and get instant emotion analysis.
-   **🎤 Speaker Identification**: Identify speakers from a pre-trained set of 122 classes.
-   **📏 Variable Length Support**: Handle audio inputs of any duration using smart segmentation (3-second sliding windows) and aggregation.
-   **📊 Interactive Visualizations**: Explore probability distributions and confidence scores with dynamic Plotly charts.
-   **📁 File Upload**: Support for WAV, MP3, OGG, and FLAC files for offline analysis.
-   **🧠 Advanced Models**:
    -   **Emotion Model**: CNN-LSTM-Attention hybrid architecture for capturing spectral and temporal features.
    -   **Speaker Model**: Deep ResNet-style CNN with Squeeze-and-Excitation (SE) attention blocks.

## 🏗️ Architecture

The system uses two specialized deep learning models defined in `models.py`:

1.  **AdvancedEmotionModel**:
    -   **Input**: Mel-spectrograms + MFCCs
    -   **Backbone**: Residual Blocks with Squeeze-and-Excitation Attention
    -   **Temporal Reasoning**: Bidirectional LSTM layers
    -   **Fusion**: Concatenates spatial (CNN) and temporal (LSTM) features

2.  **AdvancedSpeakerModel**:
    -   **Backbone**: Deep Residual Network (ResNet style)
    -   **Attention**: SE Attention blocks for channel-wise feature refinement
    -   **Head**: Large bottleneck dense layers for robust speaker embeddings

## 🚀 Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install Dependencies**:
    Ensure you have Python 3.8+ installed. Install the required packages:

    ```bash
    pip install streamlit torch numpy librosa plotly sounddevice soundfile joblib
    ```

    *Note: For GPU acceleration, install the appropriate CUDA version of PyTorch.*

## 💻 Usage

1.  **Start the Application**:
    Run the Streamlit app from your terminal:

    ```bash
    streamlit run app.py
    ```

2.  **Using the Interface**:
    -   **Emotion Detection Mode**: Switch to this mode to record or upload audio for pure emotion analysis. Use the slider to adjust recording duration (3-30 seconds).
    -   **Emotion + Speaker Analysis**: Upload a file to get a comprehensive report including both the detected emotion and the identified speaker.

## 📂 Project Structure

```
├── app.py                  # Main Streamlit application
├── models.py               # PyTorch model architectures
├── preprocessing.py        # Audio processing and inference pipelines
├── speech.ipynb            # Training and research notebook
├── Emotion_Recognition_best.pth   # Pre-trained Emotion Model weights
├── Speaker_Identification_best.pth  # Pre-trained Speaker Model weights
├── label_encoder_emotion.pkl      # Emotion label encoder
└── label_encoder_speaker.pkl      # Speaker label encoder
```

## ⚠️ Notes

-   **Recording**: Requires a functional microphone. If you encounter errors on Windows, ensure no other application is exclusively using the microphone.
-   **Input Format**: The system automatically resamples input audio to **22,050 Hz** for consistency with the training data.

---
*Built with ❤️ for SPR-PROJECT*
