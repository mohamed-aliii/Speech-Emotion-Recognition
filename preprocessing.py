"""
Preprocessing Module for Speech Emotion Recognition and Speaker Identification
Contains all necessary functions for inference and model deployment
"""

import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Advanced audio preprocessing pipeline for inference"""
    
    def __init__(self, sr=22050, duration=3.0):
        self.sr = sr
        self.duration = duration
        self.target_length = int(sr * duration)
    
    def load_audio(self, filepath):
        """Load audio with librosa"""
        try:
            audio, sr = librosa.load(filepath, sr=self.sr, duration=self.duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def trim_silence(self, audio, top_db=20):
        """Remove silence from beginning and end"""
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return audio_trimmed
        except Exception as e:
            print(f"Warning: Could not trim silence: {e}")
            return audio
    
    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def pad_or_truncate(self, audio):
        """Pad or truncate audio to target length"""
        if len(audio) < self.target_length:
            return np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        elif len(audio) > self.target_length:
            return audio[:self.target_length]
        return audio
    
    def segment_audio(self, audio):
        """Segment audio into fixed length chunks"""
        # If audio is shorter than target length, pad it
        if len(audio) <= self.target_length:
            return [self.pad_or_truncate(audio)]
        
        # Calculate number of segments
        num_segments = int(np.ceil(len(audio) / self.target_length))
        segments = []
        
        for i in range(num_segments):
            start = i * self.target_length
            end = min((i + 1) * self.target_length, len(audio))
            segment = audio[start:end]
            
            # Pad the last segment if needed
            if len(segment) < self.target_length:
                segment = np.pad(segment, (0, self.target_length - len(segment)), mode='constant')
            
            segments.append(segment)
            
        return segments

    def preprocess(self, filepath):
        """Complete preprocessing pipeline for inference"""
        audio, sr = self.load_audio(filepath)
        
        if audio is None:
            return None
        
        # Trim silence
        audio = self.trim_silence(audio)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Segment audio instead of just padding/truncating multiple times
        # Returns a list of segments
        segments = self.segment_audio(audio)
        
        return segments


class FeatureExtractor:
    """Extract audio features for emotion and speaker recognition"""
    
    def __init__(self, sr=22050):
        self.sr = sr
    
    def extract_mfcc(self, audio, n_mfcc=40):
        """Extract MFCC features"""
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            mfcc_std = np.std(mfcc.T, axis=0)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
            return np.concatenate([mfcc_mean, mfcc_std, mfcc_delta_mean])
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return None
    
    def extract_chroma(self, audio):
        """Extract chroma features"""
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
            chroma_mean = np.mean(chroma.T, axis=0)
            chroma_std = np.std(chroma.T, axis=0)
            return np.concatenate([chroma_mean, chroma_std])
        except Exception as e:
            print(f"Error extracting chroma: {e}")
            return None
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        try:
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            
            features = np.concatenate([
                [np.mean(spectral_centroid), np.std(spectral_centroid)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                np.mean(spectral_contrast.T, axis=0),
                [np.mean(zcr), np.std(zcr)]
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            return None
    
    def extract_energy(self, audio):
        """Extract energy features"""
        try:
            rms = librosa.feature.rms(y=audio)
            return np.array([np.mean(rms), np.std(rms)])
        except Exception as e:
            print(f"Error extracting energy: {e}")
            return None
    
    def extract_pitch(self, audio):
        """Extract pitch features"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr)
            pitches = pitches[pitches > 0]
            if len(pitches) > 0:
                return np.array([np.mean(pitches), np.std(pitches), np.min(pitches), np.max(pitches)])
            return np.array([0, 0, 0, 0])
        except Exception as e:
            print(f"Error extracting pitch: {e}")
            return np.array([0, 0, 0, 0])
    
    def extract_mel_spectrogram(self, audio, n_mels=128):
        """Extract mel spectrogram for deep learning models"""
        try:
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception as e:
            print(f"Error extracting mel spectrogram: {e}")
            return None
    
    def extract_all_features(self, audio):
        """Extract all traditional features and concatenate"""
        mfcc = self.extract_mfcc(audio)
        chroma = self.extract_chroma(audio)
        spectral = self.extract_spectral_features(audio)
        energy = self.extract_energy(audio)
        pitch = self.extract_pitch(audio)
        
        if None in [mfcc, chroma, spectral, energy, pitch]:
            return None
        
        return np.concatenate([mfcc, chroma, spectral, energy, pitch])
    
    def pad_feature_array(self, feature_array, target_shape):
        """Pad or crop feature array to target shape"""
        h_target, w_target = target_shape
        h_current, w_current = feature_array.shape
        
        # Pad if too small, crop if too large
        h_pad = max(0, h_target - h_current)
        w_pad = max(0, w_target - w_current)
        
        if h_pad > 0 or w_pad > 0:
            feature_array = np.pad(
                feature_array, 
                ((0, h_pad), (0, w_pad)), 
                mode='constant', 
                constant_values=0
            )
        
        # Crop if necessary
        feature_array = feature_array[:h_target, :w_target]
        
        return feature_array
    
    def extract_features_for_cnn(self, audio, n_mels=128):
        """Extract features suitable for CNN input (mel spectrogram + MFCC)"""
        try:
            mel_spec = self.extract_mel_spectrogram(audio, n_mels=n_mels)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=40)
            
            if mel_spec is None or mfcc is None:
                return None
            
            # Combine mel spectrogram and MFCC
            features = np.concatenate([mel_spec, mfcc], axis=0)
            
            return features
        except Exception as e:
            print(f"Error extracting CNN features: {e}")
            return None


class ModelLoader:
    """Load trained models and label encoders"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.emotion_model = None
        self.speaker_model = None
        self.label_encoder_emotion = None
        self.label_encoder_speaker = None
        self.target_feature_shape = None
        self.error_message = None
    
    def load_models(self, emotion_model_path, speaker_model_path, 
                   emotion_encoder_path, speaker_encoder_path,
                   emotion_model_class, speaker_model_class,
                   input_shape, num_emotions, num_speakers):
        """Load all models and encoders. Returns (success, error_message)"""
        
        try:
            # Load label encoders
            print(f"Loading emotion encoder from: {emotion_encoder_path}")
            self.label_encoder_emotion = joblib.load(emotion_encoder_path)
            print(f"✓ Emotion label encoder loaded. Classes: {self.label_encoder_emotion.classes_}")
            
            print(f"Loading speaker encoder from: {speaker_encoder_path}")
            self.label_encoder_speaker = joblib.load(speaker_encoder_path)
            print(f"✓ Speaker label encoder loaded. Number of speakers: {len(self.label_encoder_speaker.classes_)}")
            
            # Initialize and load emotion model
            print(f"\nInitializing emotion model with input_shape={input_shape}, num_emotions={num_emotions}")
            self.emotion_model = emotion_model_class(
                input_shape=input_shape,
                num_emotions=num_emotions
            )
            print(f"✓ Emotion model initialized")
            
            print(f"Loading emotion model weights from: {emotion_model_path}")
            emotion_state = torch.load(emotion_model_path, map_location=self.device)
            print(f"  Loaded state dict keys: {list(emotion_state.keys())[:5]}... (showing first 5)")
            
            self.emotion_model.load_state_dict(emotion_state)
            self.emotion_model = self.emotion_model.to(self.device)
            self.emotion_model.eval()
            print("✓ Emotion recognition model loaded successfully")
            
            # Initialize and load speaker model
            print(f"\nInitializing speaker model with input_shape={input_shape}, num_speakers={num_speakers}")
            self.speaker_model = speaker_model_class(
                input_shape=input_shape,
                num_speakers=num_speakers
            )
            print(f"✓ Speaker model initialized")
            
            print(f"Loading speaker model weights from: {speaker_model_path}")
            speaker_state = torch.load(speaker_model_path, map_location=self.device)
            print(f"  Loaded state dict keys: {list(speaker_state.keys())[:5]}... (showing first 5)")
            
            self.speaker_model.load_state_dict(speaker_state)
            self.speaker_model = self.speaker_model.to(self.device)
            self.speaker_model.eval()
            print("✓ Speaker identification model loaded successfully")
            
            print("\n✓✓✓ All models loaded successfully! ✓✓✓")
            return True
            
        except FileNotFoundError as e:
            error_msg = f"File not found error: {e}"
            print(f"❌ {error_msg}")
            self.error_message = error_msg
            return False
        except RuntimeError as e:
            error_msg = f"Runtime error (likely model architecture mismatch): {e}"
            print(f"❌ {error_msg}")
            self.error_message = error_msg
            return False
        except Exception as e:
            import traceback
            error_msg = f"Error loading models: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            self.error_message = error_msg
            return False
    
    def load_target_feature_shape(self, shape_path):
        """Load the target feature shape used during training"""
        try:
            data = joblib.load(shape_path)
            self.target_feature_shape = data
            print(f"✓ Target feature shape loaded: {self.target_feature_shape}")
            return True
        except Exception as e:
            print(f"Warning: Could not load target feature shape: {e}")
            return False


class InferencePreprocessor:
    """Complete preprocessing pipeline for inference"""
    
    def __init__(self, sr=22050, duration=3.0, device='cpu'):
        self.sr = sr
        self.duration = duration
        self.device = device
        self.audio_preprocessor = AudioPreprocessor(sr=sr, duration=duration)
        self.feature_extractor = FeatureExtractor(sr=sr)
        self.target_feature_shape = None
    
    def set_target_feature_shape(self, shape):
        """Set the target feature shape from training"""
        self.target_feature_shape = shape
    
    def preprocess_audio_file(self, filepath):
        """Preprocess a single audio file"""
        # Returns a list of segments
        segments = self.audio_preprocessor.preprocess(filepath)
        if segments is None:
            return None
        return segments
    
    def extract_and_prepare_features(self, segments):
        """Extract features and prepare for model input (batched)"""
        if segments is None:
            return None
        
        batch_features = []
        
        try:
            for audio in segments:
                # Extract CNN features
                features = self.feature_extractor.extract_features_for_cnn(audio)
                
                if features is None:
                    continue
                
                # Pad to target shape if available
                if self.target_feature_shape is not None:
                    features = self.feature_extractor.pad_feature_array(
                        features, 
                        self.target_feature_shape
                    )
                
                # Add channel dimension: (height, width) -> (1, height, width)
                features = features[np.newaxis, :, :]
                batch_features.append(features)
            
            if not batch_features:
                return None
                
            # Stack into a batch: (Batch, 1, Height, Width)
            batch_array = np.stack(batch_features)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(batch_array).to(self.device)
            
            return features_tensor
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def process_audio_file(self, filepath):
        """Complete pipeline: load -> preprocess -> extract features"""
        segments = self.preprocess_audio_file(filepath)
        if segments is None:
            return None
        
        features = self.extract_and_prepare_features(segments)
        return features


class InferenceEngine:
    """Complete inference engine for predictions"""
    
    def __init__(self, emotion_model, speaker_model, 
                 label_encoder_emotion, label_encoder_speaker, device='cpu'):
        self.emotion_model = emotion_model
        self.speaker_model = speaker_model
        self.label_encoder_emotion = label_encoder_emotion
        self.label_encoder_speaker = label_encoder_speaker
        self.device = device
    
    def predict_emotion(self, features):
        """Predict emotion from audio features (can be batched)"""
        try:
            with torch.no_grad():
                # features has shape (Batch, 1, H, W)
                outputs = self.emotion_model(features)
                probs = F.softmax(outputs, dim=1)
                
                # Aggregate predictions across batch (average probabilities)
                avg_probs = torch.mean(probs, dim=0, keepdim=True)
                
                pred_idx = torch.argmax(avg_probs, dim=1)
                pred_prob = torch.max(avg_probs, dim=1).values
            
            emotion_label = self.label_encoder_emotion.classes_[pred_idx.item()]
            confidence = pred_prob.item() * 100
            
            return emotion_label, confidence, avg_probs.cpu().numpy()
            
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return None, None, None
    
    def predict_speaker(self, features):
        """Predict speaker from audio features (can be batched)"""
        try:
            with torch.no_grad():
                outputs = self.speaker_model(features)
                probs = F.softmax(outputs, dim=1)
                
                # Aggregate predictions across batch
                avg_probs = torch.mean(probs, dim=0, keepdim=True)
                
                pred_idx = torch.argmax(avg_probs, dim=1)
                pred_prob = torch.max(avg_probs, dim=1).values
            
            speaker_label = self.label_encoder_speaker.classes_[pred_idx.item()]
            confidence = pred_prob.item() * 100
            
            return speaker_label, confidence, avg_probs.cpu().numpy()
            
        except Exception as e:
            print(f"Error predicting speaker: {e}")
            return None, None, None
    
    def predict_both(self, features):
        """Predict both emotion and speaker"""
        emotion_label, emotion_conf, emotion_probs = self.predict_emotion(features)
        speaker_label, speaker_conf, speaker_probs = self.predict_speaker(features)
        
        return {
            'emotion': emotion_label,
            'emotion_confidence': emotion_conf,
            'emotion_probabilities': emotion_probs,
            'speaker': speaker_label,
            'speaker_confidence': speaker_conf,
            'speaker_probabilities': speaker_probs
        }


def load_all_preprocessing_components(emotion_model_class, speaker_model_class,
                                     emotion_model_path='Emotion_Recognition_best.pth',
                                     speaker_model_path='Speaker_Identification_best.pth',
                                     emotion_encoder_path='label_encoder_emotion.pkl',
                                     speaker_encoder_path='label_encoder_speaker.pkl',
                                     device='cpu'):
    """
    Load all preprocessing components and models for deployment
    
    Args:
        emotion_model_class: The emotion model class (from main script)
        speaker_model_class: The speaker model class (from main script)
        emotion_model_path: Path to emotion model weights
        speaker_model_path: Path to speaker model weights
        emotion_encoder_path: Path to emotion label encoder
        speaker_encoder_path: Path to speaker label encoder
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        tuple: (preprocessor, inference_engine) ready for deployment
    """
    
    try:
        # Initialize components
        preprocessor = InferencePreprocessor(sr=22050, duration=3.0, device=device)
        model_loader = ModelLoader(device=device)
        
        # Load models (using example shapes - adjust based on your training output)
        input_shape = (168, 104)  # Adjust to match your training output
        num_emotions = 8  # Adjust based on your dataset
        num_speakers = 122  # Adjust based on your dataset
        
        success = model_loader.load_models(
            emotion_model_path=emotion_model_path,
            speaker_model_path=speaker_model_path,
            emotion_encoder_path=emotion_encoder_path,
            speaker_encoder_path=speaker_encoder_path,
            emotion_model_class=emotion_model_class,
            speaker_model_class=speaker_model_class,
            input_shape=input_shape,
            num_emotions=num_emotions,
            num_speakers=num_speakers
        )
        
        if not success:
            print("Failed to load models")
            return None, None
        
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
        
        print("\n✓ All preprocessing components and models loaded successfully!")
        return preprocessor, inference_engine
        
    except Exception as e:
        print(f"Error loading preprocessing components: {e}")
        return None, None


# Utility functions for batch processing
def preprocess_batch(audio_files, preprocessor):
    """Process multiple audio files"""
    results = []
    for filepath in audio_files:
        features = preprocessor.process_audio_file(filepath)
        if features is not None:
            results.append((filepath, features))
    return results


def predict_batch(feature_list, inference_engine):
    """Make predictions on a batch of features"""
    predictions = []
    for features in feature_list:
        result = inference_engine.predict_both(features)
        predictions.append(result)
    return predictions
