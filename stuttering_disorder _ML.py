import librosa
import pandas as pd
import numpy as np
import os, glob, time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
import joblib

warnings.filterwarnings('ignore')

# Emotion labels from filename
emotions = {
    '01': 'Parkinson',
    '02': 'Normal',
}

# Feature extraction
def get_features(file):
    try:
        waveform, sample_rate = librosa.load(file, duration=5.0)  # Limit to 5s
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40)

        chroma_mean = np.mean(chroma, axis=1)
        mel_mean = np.mean(mel, axis=1)
        mfcc_mean = np.mean(mfcc, axis=1)

        return np.hstack((chroma_mean, mel_mean, mfcc_mean))  # Shape: (180,)
    except Exception as e:
        print(f"\nError processing {file}: {e}")
        return None

# Load dataset
def load_data():
    X, y = [], []
    files = glob.glob("DataSetFinal\\Data_*\\*.wav")
    print(f"Found {len(files)} audio files.")
    start = time.time()

    for idx, file in enumerate(files):
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[-2]

        if emotion_code not in emotions:
            continue

        features = get_features(file)
        if features is not None and len(features) == 180:
            X.append(features)
            y.append(emotions[emotion_code])

        if (idx + 1) % 10 == 0:
            print(f'\rProcessed {idx + 1}/{len(files)} files in {time.time() - start:.1f}s', end='')

    print("\nData loading completed.")
    return np.array(X), np.array(y)

# Run everything
features, labels = load_data()
features_df = pd.DataFrame(features)

# Scalers
scaler_std = StandardScaler()
features_scaled = scaler_std.fit_transform(features)

scaler_minmax = MinMaxScaler()
features_minmax = scaler_minmax.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=69)
X_train_scaled = scaler_std.transform(X_train)
X_test_scaled = scaler_std.transform(X_test)
X_train_minmax = scaler_minmax.transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# Default RF
model_default = RandomForestClassifier(random_state=69)
model_default.fit(X_train, y_train)
print(f'Default RF Accuracy: {100 * model_default.score(X_test, y_test):.2f}%')

# Tuned RF
model_tuned = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    warm_start=True,
    max_features='sqrt',
    oob_score=True,
    random_state=69
)
model_tuned.fit(X_train, y_train)
print(f'Tuned RF Accuracy: {100 * model_tuned.score(X_test, y_test):.2f}%')

# Save model
joblib.dump(model_tuned, "random_forest_model.pkl")
print("Tuned model saved as 'random_forest_model.pkl'")


from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict on test set
y_pred = model_tuned.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Normal", "Parkinson"])

# Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Parkinson"]))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Parkinson"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Tuned Random Forest")
plt.show()