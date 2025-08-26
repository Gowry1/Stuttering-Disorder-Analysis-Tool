import librosa
import numpy as np
import joblib
import sys

# Load your trained Random Forest model
model_filename = "random_forest_model.pkl"
loaded_model = joblib.load(model_filename)

# Path to the input audio file you want to predict
input_audio_file = "input.wav"  # Replace with your actual file path


# Function to extract audio features
def feature_chromagram(waveform, sample_rate):
    stft_spectrogram = np.abs(librosa.stft(waveform))
    return np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T, axis=0)


def feature_melspectrogram(waveform, sample_rate):
    return np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000).T, axis=0)


def feature_mfcc(waveform, sample_rate):
    return np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T, axis=0)


def get_features(file):
    waveform, sample_rate = librosa.load(file, duration=5.0)  # Limit to 5 seconds
    if len(waveform) < 2048:
        raise ValueError(f"Audio too short for analysis: only {len(waveform)} samples.")

    chroma = feature_chromagram(waveform, sample_rate)
    mel = feature_melspectrogram(waveform, sample_rate)
    mfcc = feature_mfcc(waveform, sample_rate)

    return np.hstack((chroma, mel, mfcc))  # Shape (180,)


# Prediction
try:
    input_features = get_features(input_audio_file)
    prediction = loaded_model.predict([input_features])
    print(f"Predicted Result: {prediction[0]}")
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)
