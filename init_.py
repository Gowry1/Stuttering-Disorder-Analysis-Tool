from flask import Flask
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
bcrypt = Bcrypt()


def create_app():
    app = Flask(__name__)

    # Configure PostgreSQL connection
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/stuttering_db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.secret_key = '!das6356565h'  # optional, for sessions and flash messages

    db.init_app(app)
    bcrypt.init_app(app)


    # Import models after db is initialized
    from models.User import User
    from models.RefreshToken import RefreshToken

    # Create all tables
    with app.app_context():
        db.create_all()

    return app
