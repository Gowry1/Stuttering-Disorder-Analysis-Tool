from init_ import db

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)  # Example: "Male", "Female", "Other"

    created_at = db.Column(db.DateTime, server_default=db.func.now())

    results = db.relationship('Result', backref='user', lazy=True)
