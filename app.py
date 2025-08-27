import traceback
from functools import wraps

from flask import Flask, render_template, request, jsonify
import re
from datetime import datetime
import time

import jwt  # ✅ this is PyJWT

from utils.jwt_utils import JWTManager

# Optional imports for ML functionality
try:
    from pydub import AudioSegment, silence
    import librosa
    import soundfile as sf
    import numpy as np
    import joblib
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML dependencies not available: {e}")
    print("Running in basic mode without audio processing capabilities")
    ML_AVAILABLE = False

# Optional imports for audio recording
try:
    import pyaudio
    import wave
    AUDIO_RECORDING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio recording dependencies not available: {e}")
    print("Audio recording functionality disabled")
    AUDIO_RECORDING_AVAILABLE = False

import init_

from flask import Flask, render_template, request, redirect, flash, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import flash, redirect, render_template, request, url_for
from flask import Blueprint, request, jsonify
from models.Result import Result, DiseaseStatusEnum
from init_ import db

from auth import auth_bp
from models.User import User

app = Flask(__name__)
app.secret_key = '!das6356565h'
app = init_.create_app()
CORS(app)
app.register_blueprint(auth_bp)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root@localhost/stutteringdisorder'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)
# login_manager = LoginManager(app)
#
# class Users(db.Model, UserMixin):
#     __tablename__ = 'users'
#
#     UserID = db.Column(db.Integer, primary_key=True)
#     Username = db.Column(db.String(255), unique=True, nullable=False)
#     Password = db.Column(db.String(255), nullable=False)
#     Email = db.Column(db.String(255), unique=True, nullable=False)
#     RegistrationDate = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow)
#
#     def check_password(self, password):
#         return check_password_hash(self.Password, password)
#
# @login_manager.user_loader
# def load_user(user_id):
#     return Users.query.get(int(user_id))
#
# @app.route('/', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form.get('Username')
#         password = request.form.get('Password')
#
#         user = Users.query.filter_by(Username=username).first()
#
#         if user and user.check_password(password):
#        #     login_user(user)
#             flash('Login successful!', 'success')
#             return redirect(url_for('LiveMaster'))
#         else:
#             flash('Login failed. Please check your username and password.', 'error')
#
#     return render_template('login.html')
#
# @app.route('/registration', methods=['GET', 'POST'])
# def registration():
#     if request.method == 'POST':
#         try:
#             username = request.form.get('Username')
#             password = request.form.get('Password')
#             email = request.form.get('Email')
#
#             # Check if any field is empty
#             if not username or not password or not email:
#                 flash('Please fill in all fields.', 'error')
#                 return redirect(url_for('registration'))
#
#             # Check password strength
#             if len(password) < 8:
#                 flash('Password must be at least 8 characters long.', 'error')
#                 return redirect(url_for('registration'))
#             elif not re.match(r'^(?=.*[A-Za-z])(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', password):
#                 flash('Password must contain at least one letter and one symbol.', 'error')
#                 return redirect(url_for('registration'))
#
#             # Check if username is unique
#             existing_user = Users.query.filter_by(Username=username).first()
#             if existing_user:
#                 flash('Username already exists. Please choose a different username.', 'error')
#                 return redirect(url_for('registration'))
#
#             # Check if email is unique
#             existing_email = Users.query.filter_by(Email=email).first()
#             if existing_email:
#                 flash('Email address already exists. Please use a different email.', 'error')
#                 return redirect(url_for('registration'))
#
#             # Hash the password
#             hashed_password = generate_password_hash(password, method='sha256')
#
#             user_data = {
#                 'Username': username,
#                 'Password': hashed_password,
#                 'Email': email,
#                 'RegistrationDate': datetime.now(),
#             }
#
#             new_user = Users(**user_data)
#             db.session.add(new_user)
#             db.session.commit()
#             flash('Registered successfully!', 'success')
#             return redirect(url_for('registration'))
#
#         except Exception as e:
#             db.session.rollback()  # Roll back the session to prevent saving erroneous data
#             flash(f'Error registering user: {e}', 'error')
#
#     return render_template('registration.html')
#
#


# Live Master-------------------------------------------------------------------------------------
# Additional imports are handled at the top of the file with optional imports
import os
import tempfile
import threading

# Global variables for recording state
recording_state = {
    "is_recording": False,
    "start_time": None,
    "duration": 8,
    "recording_thread": None,
    "audio_stream": None,
    "audio_frames": []
}

tmp_dir = tempfile.gettempdir()
input_filename = os.path.join(app.root_path, "input.wav")
mono_output_filename = os.path.join(app.root_path, "static/mono_output.wav")


def start_audio_recording(sample_rate=44100, channels=1):
    """Start audio recording and return stream object"""
    if not AUDIO_RECORDING_AVAILABLE:
        raise Exception("Audio recording dependencies not available. Please install pyaudio.")

    audio = pyaudio.PyAudio()
    format = pyaudio.paInt16
    chunk = 1024

    # Open a new audio stream
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk)

    print("Recording started...")
    return audio, stream, format, chunk, sample_rate, channels


def stop_audio_recording_and_save(audio, stream, format, chunk, sample_rate, channels, frames, output_filename):
    """Stop audio recording and save to file"""
    print("Stopping recording...")

    # Close and terminate the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Recording saved to {output_filename}")
    return output_filename


def record_audio(output_filename, duration_seconds=3, sample_rate=44100, channels=1):
    """Legacy function for backward compatibility - records for fixed duration"""
    if not AUDIO_RECORDING_AVAILABLE:
        raise Exception("Audio recording dependencies not available. Please install pyaudio.")

    audio = pyaudio.PyAudio()

    # Define the audio settings
    format = pyaudio.paInt16
    chunk = 1024

    # Open a new audio stream
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk)

    print("Recording...")

    frames = []

    # Record audio for the specified duration
    for _ in range(0, int(sample_rate / chunk * duration_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    # Close and terminate the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    return output_filename  # Return the saved file path


def convert_to_mono(input_filename, output_filename):
    audio = AudioSegment.from_wav(input_filename)
    mono_audio = audio.set_channels(1)
    mono_audio.export(output_filename, format="wav")


def feature_chromagram(waveform, sample_rate):
    # STFT
    stft_spectrogram = np.abs(librosa.stft(waveform))
    chromagram = np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T, axis=0)
    return chromagram


def feature_melspectrogram(waveform, sample_rate):
    # mel spectrogram
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000).T,
                             axis=0)
    return melspectrogram


def feature_mfcc(waveform, sample_rate):
    # MFCCs
    mfc_coefficients = np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfc_coefficients


def get_features(file):
    if not ML_AVAILABLE:
        raise Exception("ML dependencies not available. Please install librosa, soundfile, and numpy.")

    # load an individual sound file
    with sf.SoundFile(file) as audio:
        waveform = audio.read(dtype="float32")
        sample_rate = audio.samplerate
        # compute features of sound file
        chromagram = feature_chromagram(waveform, sample_rate)
        melspectrogram = feature_melspectrogram(waveform, sample_rate)
        mfc_coefficients = feature_mfcc(waveform, sample_rate)
        feature_matrix = np.array([])
        feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))
        return feature_matrix


@app.route('/system')
def LiveMaster():
    selected_message = 'Say:We are studying from last 2 hours '
    return render_template('LiveMaster.html', prediction=None, selected_message=selected_message)


@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording_state

    try:
        # Check if already recording
        if recording_state["is_recording"]:
            return jsonify({
                "status": "error",
                "message": "Recording is already in progress"
            }), 400

        if not AUDIO_RECORDING_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "Audio recording dependencies not available. Please install pyaudio."
            }), 500

        # Start the audio recording
        audio, stream, format, chunk, sample_rate, channels = start_audio_recording()

        # Update recording state
        recording_state.update({
            "is_recording": True,
            "start_time": time.time(),
            "audio_stream": {
                "audio": audio,
                "stream": stream,
                "format": format,
                "chunk": chunk,
                "sample_rate": sample_rate,
                "channels": channels
            },
            "audio_frames": []
        })

        # Start a background thread to collect audio data
        def collect_audio():
            try:
                while recording_state["is_recording"]:
                    if recording_state["audio_stream"]:
                        data = recording_state["audio_stream"]["stream"].read(
                            recording_state["audio_stream"]["chunk"],
                            exception_on_overflow=False
                        )
                        recording_state["audio_frames"].append(data)

                        # Auto-stop after 8 seconds
                        if time.time() - recording_state["start_time"] >= 8:
                            recording_state["is_recording"] = False
                            break
            except Exception as e:
                print(f"Error in audio collection: {e}")
                recording_state["is_recording"] = False

        recording_thread = threading.Thread(target=collect_audio)
        recording_thread.daemon = True
        recording_thread.start()
        recording_state["recording_thread"] = recording_thread

        return jsonify({
            "status": "recording_started",
            "message": "Recording started successfully",
            "recording_id": f"rec_{int(time.time())}"
        })

    except Exception as e:
        # Reset recording state on error
        recording_state.update({
            "is_recording": False,
            "start_time": None,
            "audio_stream": None,
            "audio_frames": []
        })
        return jsonify({
            "status": "error",
            "message": f"Error starting recording: {str(e)}"
        }), 500




# Define the list of predefined messages
messages = [
    "Say: I am going",
    "Say: Today it will not be possible to do",
    "Say: We are waiting for you",
    "Say: I went to market by yesterday evening with my friends"
]

# Initialize a global variable to keep track of the index of the next message to display
current_message_index = 0

from flask import jsonify

def save_prediction_to_database(user_id, prediction, recording_duration=8.0):
    """Save prediction result to database and return statistics"""
    try:
        # Map prediction to enum
        disease_status_str = prediction.upper()
        if disease_status_str not in DiseaseStatusEnum.__members__:
            disease_status_str = "NORMAL"  # Default fallback

        disease_status = DiseaseStatusEnum[disease_status_str]

        # Create new result
        result = Result(
            user_id=user_id,
            disease_status=disease_status,
            recording_duration=recording_duration
        )
        db.session.add(result)
        db.session.commit()

        return result.id
    except Exception as e:
        db.session.rollback()
        print(f"Error saving prediction to database: {e}")
        return None


def get_user_statistics(user_id):
    """Get user's prediction statistics from database"""
    try:
        # Get all results for the user
        user_results = Result.query.filter_by(user_id=user_id).all()

        if not user_results:
            return {
                "normal_count": 0,
                "total_predictions": 0,
                "percentage_normal": 0
            }

        # Count normal predictions
        normal_count = sum(1 for result in user_results
                          if result.disease_status == DiseaseStatusEnum.NORMAL)
        total_predictions = len(user_results)
        percentage_normal = int((normal_count / total_predictions) * 100) if total_predictions > 0 else 0

        return {
            "normal_count": normal_count,
            "total_predictions": total_predictions,
            "percentage_normal": percentage_normal
        }
    except Exception as e:
        print(f"Error getting user statistics: {e}")
        return {
            "normal_count": 0,
            "total_predictions": 0,
            "percentage_normal": 0
        }


def get_global_statistics():
    """Get global prediction statistics from database (for unauthenticated users)"""
    try:
        # Get all results from all users
        all_results = Result.query.all()

        if not all_results:
            return {
                "normal_count": 0,
                "total_predictions": 0,
                "percentage_normal": 0
            }

        # Count normal predictions
        normal_count = sum(1 for result in all_results
                          if result.disease_status == DiseaseStatusEnum.NORMAL)
        total_predictions = len(all_results)
        percentage_normal = int((normal_count / total_predictions) * 100) if total_predictions > 0 else 0

        return {
            "normal_count": normal_count,
            "total_predictions": total_predictions,
            "percentage_normal": percentage_normal
        }
    except Exception as e:
        print(f"Error getting global statistics: {e}")
        return {
            "normal_count": 0,
            "total_predictions": 0,
            "percentage_normal": 0
        }


def preprocess_audio(input_path, output_path, target_rate=16000):
    if not ML_AVAILABLE:
        raise Exception("Audio processing dependencies not available. Please install pydub.")

    audio = AudioSegment.from_wav(input_path)
    audio = audio.set_channels(1)

    # Trim silence
    nonsilent_ranges = silence.detect_nonsilent(audio, silence_thresh=-40, min_silence_len=300)
    if nonsilent_ranges:
        start = nonsilent_ranges[0][0]
        end = nonsilent_ranges[-1][1]
        audio = audio[start:end]

    # Normalize volume
    change_dBFS = -20.0 - audio.dBFS
    audio = audio.apply_gain(change_dBFS)

    # Resample
    audio = audio.set_frame_rate(target_rate)
    audio.export(output_path, format="wav")

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global current_message_index, recording_state

    try:
        # Check if we have any recording data (either in progress or completed)
        has_recording_data = (
            recording_state.get("is_recording", False) or
            recording_state.get("audio_frames", []) or
            recording_state.get("recording_thread") is not None
        )

        if not has_recording_data:
            return jsonify({
                "status": "error",
                "message": "No recording in progress or no audio data available"
            }), 400

        # Stop recording if still in progress
        recording_state["is_recording"] = False

        # Wait for recording thread to complete if it exists
        if recording_state["recording_thread"] and recording_state["recording_thread"].is_alive():
            recording_state["recording_thread"].join(timeout=3)  # Wait up to 3 seconds

        # Save the recorded audio
        if recording_state.get("audio_stream") and recording_state.get("audio_frames"):
            audio_info = recording_state["audio_stream"]
            stop_audio_recording_and_save(
                audio_info["audio"],
                audio_info["stream"],
                audio_info["format"],
                audio_info["chunk"],
                audio_info["sample_rate"],
                audio_info["channels"],
                recording_state["audio_frames"],
                input_filename
            )
        else:
            return jsonify({
                "status": "error",
                "message": "No audio data recorded"
            }), 400

        # Reset recording state
        recording_state.update({
            "is_recording": False,
            "start_time": None,
            "recording_thread": None,
            "audio_stream": None,
            "audio_frames": []
        })

        # Get user from token for database operations
        user_id = None
        try:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                access_token = auth_header.split(' ')[1]
                current_user = JWTManager.get_user_from_token(access_token)
                if current_user:
                    user_id = current_user.id
        except Exception as e:
            print(f"Error getting user from token: {e}")

        # Handle ML processing with fallback for compatibility issues
        prediction = "NORMAL"  # Default fallback prediction

        if ML_AVAILABLE and os.path.exists(input_filename):
            try:
                # Preprocess (mono + trim + normalize + resample)
                final_path = os.path.join(app.root_path, "static/final_processed.wav")
                preprocess_audio(input_filename, final_path)

                # Extract features
                features = get_features(final_path)

                # Try to use the loaded model if available
                if loaded_model is not None:
                    # ✅ Load and apply scaler
                    scaler_path = os.path.join(app.root_path, "scaler.pkl")
                    if os.path.exists(scaler_path):
                        try:
                            scaler = joblib.load(scaler_path)
                            features = scaler.transform([features])
                        except Exception as e:
                            print(f"Warning: Scaler compatibility issue: {e}")
                            features = [features]  # fallback: unscaled
                    else:
                        features = [features]  # fallback: unscaled

                    prediction = loaded_model.predict(features)[0]
                    print(f"✅ ML prediction successful: {prediction}")
                else:
                    print("⚠️ Using fallback prediction due to model compatibility issues")

            except Exception as e:
                print(f"⚠️ ML processing failed, using fallback: {e}")
                prediction = "NORMAL"
        else:
            print("⚠️ ML not available or no audio file, using fallback prediction")
            prediction = "NORMAL"

        # Message rotation
        selected_message = messages[current_message_index]
        current_message_index = (current_message_index + 1) % len(messages)

        # Save prediction to database and get statistics
        if user_id:
            save_prediction_to_database(user_id, prediction)
            stats = get_user_statistics(user_id)
        else:
            # For unauthenticated users, use global statistics
            stats = get_global_statistics()

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "selected_message": selected_message,
            "normal_count": stats["normal_count"],
            "percentage_normal": stats["percentage_normal"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Exception: {str(e)}"}), 500


# upload test
# Load your trained Random Forest model (only if ML is available)
model_filename = "random_forest_model.pkl"
loaded_model = None
if ML_AVAILABLE:
    try:
        loaded_model = joblib.load(model_filename)
        print(f"✅ Model loaded successfully from {model_filename}")
        print(f"   Model type: {type(loaded_model)}")
    except FileNotFoundError:
        print(f"⚠️ Warning: Model file {model_filename} not found")
        loaded_model = None
    except Exception as e:
        # Only set to None for critical errors, not warnings
        print(f"⚠️ Warning during model loading: {e}")
        # Try to load anyway - many warnings are non-critical
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded_model = joblib.load(model_filename)
                print(f"✅ Model loaded successfully despite warnings")
        except Exception as critical_error:
            print(f"❌ Critical error: Could not load model: {critical_error}")
            loaded_model = None

# Define the emotion labels
emotions = ['Normal', 'Stuttering_Disorder']


# Function to extract audio features
def feature_chromagram(waveform, sample_rate):
    if not ML_AVAILABLE:
        raise Exception("ML dependencies not available")
    # STFT
    stft_spectrogram = np.abs(librosa.stft(waveform))
    chromagram = np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T, axis=0)
    return chromagram


def feature_melspectrogram(waveform, sample_rate):
    if not ML_AVAILABLE:
        raise Exception("ML dependencies not available")
    # mel spectrogram
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000).T,
                             axis=0)
    return melspectrogram


def feature_mfcc(waveform, sample_rate):
    if not ML_AVAILABLE:
        raise Exception("ML dependencies not available")
    # MFCCs
    mfc_coefficients = np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfc_coefficients


def get_features(file):
    with sf.SoundFile(file) as audio:
        waveform = audio.read(dtype="float32")
        sample_rate = audio.samplerate

        # Compute features
        chromagram = feature_chromagram(waveform, sample_rate)
        melspectrogram = feature_melspectrogram(waveform, sample_rate)
        mfc_coefficients = feature_mfcc(waveform, sample_rate)

        # Concatenate the features along axis 1
        feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))

    return feature_matrix


@app.route('/upload', methods=['GET', 'POST'])
def index():
    predicted_emotion = None
    normal_count = 0  # Initialize normal_count with a default value
    total_words = 0  # Initialize total_words with a default value
    percentage_normal = 0

    if request.method == 'POST':
        print(11)

    return render_template('upload.html', predicted_emotion=predicted_emotion, normal_count=normal_count,
                           total_words=total_words, percentage_normal=percentage_normal)


# Duplicate functions removed - already defined above


#-------------------------------------------------------------------------------------------------------------------
app.config[
    'SECRET_KEY'] = '0c4b66a7305727e0a4571c745ef9768bb472b01752d3fe71468c8c0f3fb0f8643c7effe087dc4e24c3893dd69dabe048'


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            parts = request.headers['Authorization'].split()
            if len(parts) == 2 and parts[0] == 'Bearer':
                token = parts[1]

        if not token:
            return jsonify({'message': 'Access token is missing'}), 401

        try:
            # Import here to avoid circular imports
            from utils.jwt_utils import JWTManager

            # Decode and validate access token
            payload = JWTManager.decode_access_token(token)

            if not payload:
                return jsonify({'message': 'Invalid token'}), 401

            if 'error' in payload:
                return jsonify({'message': payload['error']}), 401

            # Get user from token
            current_user = JWTManager.get_user_from_token(token)

            if current_user is None:
                return jsonify({'message': 'User not found or token invalid'}), 401

        except Exception as e:
            return jsonify({'message': f'Token validation error: {str(e)}'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


# Protected route example - you can add more protected routes here
# @app.route('/protected-example', methods=['GET'])
# @token_required
# def protected_example(current_user):
#     return jsonify({
#         'message': f'Hello {current_user.username}!',
#         'user_id': current_user.id
#     })







if __name__ == '__main__':
    app.run(debug=True)
