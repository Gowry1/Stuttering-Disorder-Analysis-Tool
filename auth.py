from flask import Blueprint, request, jsonify, current_app
from flask_bcrypt import check_password_hash

from init_ import db, bcrypt

from models.Result import DiseaseStatusEnum, Result
from models.User import User
from utils.jwt_utils import JWTManager




auth_bp = Blueprint('auth', __name__)

# Register
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    # Extract fields
    full_name = data.get('full_name')
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    age = data.get('age')
    gender = data.get('gender')

    # Validate required fields
    if not all([full_name, username, email, password, age, gender]):
        return jsonify({'message': 'All fields (full_name, username, email, password, age, gender) are required'}), 400

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    # Create new user
    new_user = User(
        full_name=full_name,
        username=username,
        email=email,
        password=hashed_password,
        age=age,
        gender=gender
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201
# Login

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()

        if not data or 'email' not in data or 'password' not in data:
            return jsonify({'message': 'Missing email or password'}), 400

        user = User.query.filter_by(email=data['email']).first()

        if user and check_password_hash(user.password, data['password']):
            # Generate token pair using new JWT system
            tokens = JWTManager.generate_token_pair(user.id)

            return jsonify({
                'message': 'Login successful',
                'access_token': tokens['access_token'],
                'refresh_token': tokens['refresh_token'],
                'access_token_expires_in': tokens['access_token_expires_in'],
                'refresh_token_expires_in': tokens['refresh_token_expires_in'],
                'token_type': tokens['token_type'],
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'username': user.username,
                    'full_name': user.full_name,
                    'age': user.age,
                    'gender': user.gender
                }
            }), 200

        return jsonify({'message': 'Invalid credentials'}), 401

    except Exception as e:
        print("Login error:", e)
        return jsonify({'message': 'Server error'}), 500


# Token refresh endpoint
@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    try:
        data = request.get_json()

        if not data or 'refresh_token' not in data:
            return jsonify({'message': 'Refresh token is required'}), 400

        refresh_token = data['refresh_token']

        # Generate new access token
        result = JWTManager.refresh_access_token(refresh_token)

        if 'error' in result:
            return jsonify({'message': result['error']}), 401

        return jsonify({
            'message': 'Token refreshed successfully',
            'access_token': result['access_token'],
            'access_token_expires_in': result['access_token_expires_in'],
            'token_type': result['token_type']
        }), 200

    except Exception as e:
        print("Token refresh error:", e)
        return jsonify({'message': 'Server error'}), 500


# Logout endpoint
@auth_bp.route('/logout', methods=['POST'])
def logout():
    try:
        data = request.get_json()

        # Get refresh token from request body
        refresh_token = data.get('refresh_token') if data else None

        # Get access token from Authorization header
        auth_header = request.headers.get('Authorization')
        access_token = None
        if auth_header and auth_header.startswith('Bearer '):
            access_token = auth_header.split(' ')[1]

        if not refresh_token and not access_token:
            return jsonify({'message': 'No token provided'}), 400

        # If refresh token provided, revoke it
        if refresh_token:
            JWTManager.revoke_refresh_token(refresh_token)

        # If access token provided, get user and revoke all their tokens
        if access_token:
            user = JWTManager.get_user_from_token(access_token)
            if user:
                JWTManager.revoke_all_user_tokens(user.id)

        return jsonify({'message': 'Logged out successfully'}), 200

    except Exception as e:
        print("Logout error:", e)
        return jsonify({'message': 'Server error'}), 500


# Logout from all devices
@auth_bp.route('/logout-all', methods=['POST'])
def logout_all():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Access token is required'}), 401

        access_token = auth_header.split(' ')[1]
        user = JWTManager.get_user_from_token(access_token)

        if not user:
            return jsonify({'message': 'Invalid or expired token'}), 401

        # Revoke all refresh tokens for the user
        revoked_count = JWTManager.revoke_all_user_tokens(user.id)

        return jsonify({
            'message': f'Logged out from all devices successfully',
            'revoked_tokens': revoked_count
        }), 200

    except Exception as e:
        print("Logout all error:", e)
        return jsonify({'message': 'Server error'}), 500


# Token validation endpoint
@auth_bp.route('/validate-token', methods=['POST'])
def validate_token():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Access token is required', 'valid': False}), 401

        access_token = auth_header.split(' ')[1]
        payload = JWTManager.decode_access_token(access_token)

        if not payload or 'error' in payload:
            error_msg = payload.get('error', 'Invalid token') if payload else 'Invalid token'
            return jsonify({'message': error_msg, 'valid': False}), 401

        user = JWTManager.get_user_from_token(access_token)
        if not user:
            return jsonify({'message': 'User not found', 'valid': False}), 401

        return jsonify({
            'message': 'Token is valid',
            'valid': True,
            'user': {
                'id': user.id,
                'email': user.email,
                'username': user.username,
                'full_name': user.full_name
            }
        }), 200

    except Exception as e:
        print("Token validation error:", e)
        return jsonify({'message': 'Server error', 'valid': False}), 500


# Get current user info
@auth_bp.route('/me', methods=['GET'])
def get_current_user():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Access token is required'}), 401

        access_token = auth_header.split(' ')[1]
        user = JWTManager.get_user_from_token(access_token)

        if not user:
            return jsonify({'message': 'Invalid or expired token'}), 401

        return jsonify({
            'user': {
                'id': user.id,
                'email': user.email,
                'username': user.username,
                'full_name': user.full_name,
                'age': user.age,
                'gender': user.gender,
                'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S') if user.created_at else None
            }
        }), 200

    except Exception as e:
        print("Get current user error:", e)
        return jsonify({'message': 'Server error'}), 500


# Clean up expired tokens (admin endpoint)
@auth_bp.route('/cleanup-tokens', methods=['POST'])
def cleanup_expired_tokens():
    try:
        # This could be protected with admin authentication
        cleaned_count = JWTManager.cleanup_expired_tokens()

        return jsonify({
            'message': 'Token cleanup completed',
            'cleaned_tokens': cleaned_count
        }), 200

    except Exception as e:
        print("Token cleanup error:", e)
        return jsonify({'message': 'Server error'}), 500




@auth_bp.route('/results', methods=['POST'])
def save_result():
    try:
        # Get the current user from the token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Access token is required'}), 401

        access_token = auth_header.split(' ')[1]
        current_user = JWTManager.get_user_from_token(access_token)

        if not current_user:
            return jsonify({'message': 'Invalid or expired token'}), 401

        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400

        # Get user_id from authenticated user (more secure)
        user_id = current_user.id

        disease_status_str = data.get('disease_status', '').upper()
        if not disease_status_str:
            return jsonify({"error": "disease_status is required"}), 400
        if disease_status_str not in DiseaseStatusEnum.__members__:
            return jsonify({"error": "Invalid disease status"}), 400

        confidence_score = data.get('percentage_normal', None)  # Optional
        recording_duration = data.get('recording_duration', None)  # Optional

        disease_status = DiseaseStatusEnum[disease_status_str]

        result = Result(
            user_id=user_id,
            disease_status=disease_status,
            confidence_score=confidence_score,
            recording_duration=recording_duration
        )
        db.session.add(result)
        db.session.commit()

        return jsonify({
            "message": "Prediction result saved successfully",
            "result_id": result.id,
            "user_id": result.user_id,
            "disease_status": result.disease_status.value,
            "confidence_score": result.confidence_score,
            "created_at": result.created_at.strftime('%Y-%m-%d %H:%M:%S') if result.created_at else None
        }), 201

    except Exception as e:
        db.session.rollback()
        print("Save result error:", e)
        return jsonify({"error": "Server error"}), 500


# Get current user's results
@auth_bp.route('/my-results', methods=['GET'])
def get_my_results():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Access token is required'}), 401

        access_token = auth_header.split(' ')[1]
        current_user = JWTManager.get_user_from_token(access_token)

        if not current_user:
            return jsonify({'message': 'Invalid or expired token'}), 401

        result_history = [
            {
                "id": r.id,
                "disease_status": r.disease_status.value,
                "confidence_score": r.confidence_score,
                "recording_duration": getattr(r, 'recording_duration', None),
                "audio_file_path": getattr(r, 'audio_file_path', None),
                "created_at": r.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
            for r in current_user.results
        ]

        return jsonify({
            "user": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "full_name": current_user.full_name,
                "age": current_user.age,
                "gender": current_user.gender
            },
            "results": result_history
        })

    except Exception as e:
        print("Get my results error:", e)
        return jsonify({'message': 'Server error'}), 500