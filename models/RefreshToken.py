# models/RefreshToken.py

from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean
from datetime import datetime, timedelta
from init_ import db
import secrets

class RefreshToken(db.Model):
    __tablename__ = 'refresh_tokens'

    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(255), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_revoked = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to User
    user = db.relationship('User', backref='refresh_tokens')
    
    def __init__(self, user_id, expires_in_days=30):
        self.user_id = user_id
        self.token = secrets.token_urlsafe(64)
        self.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        self.is_revoked = False
    
    def is_expired(self):
        """Check if the refresh token is expired"""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self):
        """Check if the refresh token is valid (not expired and not revoked)"""
        return not self.is_expired() and not self.is_revoked
    
    def revoke(self):
        """Revoke the refresh token"""
        self.is_revoked = True
    
    @classmethod
    def find_by_token(cls, token):
        """Find a refresh token by its token value"""
        return cls.query.filter_by(token=token).first()
    
    @classmethod
    def revoke_all_user_tokens(cls, user_id):
        """Revoke all refresh tokens for a specific user"""
        tokens = cls.query.filter_by(user_id=user_id, is_revoked=False).all()
        for token in tokens:
            token.revoke()
        return len(tokens)
    
    @classmethod
    def cleanup_expired_tokens(cls):
        """Remove expired tokens from the database"""
        expired_tokens = cls.query.filter(cls.expires_at < datetime.utcnow()).all()
        for token in expired_tokens:
            db.session.delete(token)
        return len(expired_tokens)
