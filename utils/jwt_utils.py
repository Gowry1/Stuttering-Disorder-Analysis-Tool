# utils/jwt_utils.py

import jwt
from datetime import datetime, timedelta
from flask import current_app
from models.User import User
from models.RefreshToken import RefreshToken
from init_ import db

class JWTManager:
    """JWT Token Management Utility Class"""
    
    @staticmethod
    def generate_access_token(user_id, expires_in_minutes=15):
        """
        Generate an access token for a user
        
        Args:
            user_id (int): User ID
            expires_in_minutes (int): Token expiration time in minutes (default: 15)
            
        Returns:
            str: JWT access token
        """
        payload = {
            'user_id': user_id,
            'type': 'access',
            'exp': datetime.utcnow() + timedelta(minutes=expires_in_minutes),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(
            payload,
            current_app.config['SECRET_KEY'],
            algorithm='HS256'
        )
    
    @staticmethod
    def generate_refresh_token(user_id, expires_in_days=30):
        """
        Generate a refresh token for a user and store it in database
        
        Args:
            user_id (int): User ID
            expires_in_days (int): Token expiration time in days (default: 30)
            
        Returns:
            RefreshToken: RefreshToken model instance
        """
        # Create refresh token in database
        refresh_token = RefreshToken(user_id=user_id, expires_in_days=expires_in_days)
        db.session.add(refresh_token)
        db.session.commit()
        
        return refresh_token
    
    @staticmethod
    def generate_token_pair(user_id):
        """
        Generate both access and refresh tokens for a user
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: Dictionary containing access_token and refresh_token
        """
        access_token = JWTManager.generate_access_token(user_id)
        refresh_token = JWTManager.generate_refresh_token(user_id)
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token.token,
            'access_token_expires_in': 15 * 60,  # 15 minutes in seconds
            'refresh_token_expires_in': 30 * 24 * 60 * 60,  # 30 days in seconds
            'token_type': 'Bearer'
        }
    
    @staticmethod
    def decode_access_token(token):
        """
        Decode and validate an access token
        
        Args:
            token (str): JWT access token
            
        Returns:
            dict: Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            
            # Verify token type
            if payload.get('type') != 'access':
                return None
                
            return payload
            
        except jwt.ExpiredSignatureError:
            return {'error': 'Token has expired'}
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}
        except Exception as e:
            return {'error': f'Token error: {str(e)}'}
    
    @staticmethod
    def validate_refresh_token(token):
        """
        Validate a refresh token
        
        Args:
            token (str): Refresh token string
            
        Returns:
            RefreshToken: RefreshToken model instance or None if invalid
        """
        refresh_token = RefreshToken.find_by_token(token)
        
        if not refresh_token:
            return None
            
        if not refresh_token.is_valid():
            return None
            
        return refresh_token
    
    @staticmethod
    def refresh_access_token(refresh_token_string):
        """
        Generate a new access token using a refresh token
        
        Args:
            refresh_token_string (str): Refresh token string
            
        Returns:
            dict: New token pair or error message
        """
        refresh_token = JWTManager.validate_refresh_token(refresh_token_string)
        
        if not refresh_token:
            return {'error': 'Invalid or expired refresh token'}
        
        # Generate new access token
        access_token = JWTManager.generate_access_token(refresh_token.user_id)
        
        return {
            'access_token': access_token,
            'access_token_expires_in': 15 * 60,  # 15 minutes in seconds
            'token_type': 'Bearer'
        }
    
    @staticmethod
    def revoke_refresh_token(token):
        """
        Revoke a specific refresh token
        
        Args:
            token (str): Refresh token string
            
        Returns:
            bool: True if revoked successfully, False otherwise
        """
        refresh_token = RefreshToken.find_by_token(token)
        
        if refresh_token:
            refresh_token.revoke()
            db.session.commit()
            return True
            
        return False
    
    @staticmethod
    def revoke_all_user_tokens(user_id):
        """
        Revoke all refresh tokens for a user
        
        Args:
            user_id (int): User ID
            
        Returns:
            int: Number of tokens revoked
        """
        count = RefreshToken.revoke_all_user_tokens(user_id)
        db.session.commit()
        return count
    
    @staticmethod
    def get_user_from_token(token):
        """
        Get user object from access token
        
        Args:
            token (str): JWT access token
            
        Returns:
            User: User model instance or None if invalid
        """
        payload = JWTManager.decode_access_token(token)
        
        if not payload or 'error' in payload:
            return None
            
        user_id = payload.get('user_id')
        if not user_id:
            return None
            
        return User.query.get(user_id)
    
    @staticmethod
    def cleanup_expired_tokens():
        """
        Clean up expired refresh tokens from database
        
        Returns:
            int: Number of tokens cleaned up
        """
        count = RefreshToken.cleanup_expired_tokens()
        db.session.commit()
        return count
