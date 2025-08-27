# models/Result.py

from sqlalchemy import Column, String, Float, Enum, DateTime, ForeignKey
from datetime import datetime
from init_ import db
import enum

class DiseaseStatusEnum(enum.Enum):
    NORMAL = "NORMAL"
    PARKINSON = "PARKINSON"

class Result(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    disease_status = db.Column(db.Enum(DiseaseStatusEnum), nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    recording_duration = db.Column(db.Float, nullable=True)  # Duration in seconds
    audio_file_path = db.Column(db.String(255), nullable=True)  # Path to audio file
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to User is defined in User model with backref

    def to_dict(self):
        """Convert result to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'disease_status': self.disease_status.value,
            'confidence_score': self.confidence_score,
            'recording_duration': self.recording_duration,
            'audio_file_path': self.audio_file_path,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }
