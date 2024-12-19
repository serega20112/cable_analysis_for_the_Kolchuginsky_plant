from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///messenger.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    sender = db.relationship('User', foreign_keys=[sender_id])
    receiver = db.relationship('User', foreign_keys=[receiver_id])

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"ошибка": "Все поля обязательны!"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    user = User(username=username, email=email, password=hashed_password)
    try:
        db.session.add(user)
        db.session.commit()
        return jsonify({"сообщение": "Пользователь успешно зарегистрирован! Ожидайте подтверждения директора."}), 201
    except Exception as e:
        return jsonify({"ошибка": "Ошибка регистрации. Возможно, имя пользователя или email уже используются."}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"ошибка": "Все поля обязательны!"}), 400

    user = User.query.filter_by(username=username).first()

    if user and bcrypt.check_password_hash(user.password, password):
        if user.status != 'approved':
            return jsonify({"ошибка": "Ваша учетная запись еще не подтверждена."}), 403

        access_token = create_access_token(identity={'id': user.id, 'username': user.username, 'email': user.email})
        return jsonify({"сообщение": "Успешный вход!", "token": access_token}), 200

    return jsonify({"ошибка": "Неверное имя пользователя или пароль."}), 401

@app.route('/workers', methods=['GET'])
@jwt_required()
def get_workers():
    users = User.query.filter_by(status='approved').all()
    users_list = [{"id": user.id, "username": user.username, "email": user.email} for user in users]
    return render_template('workers.html', workers=users_list)

@app.route('/messages', methods=['POST'])
@jwt_required()
def send_message():
    data = request.get_json()
    sender_id = get_jwt_identity()['id']
    receiver_id = data.get('receiver_id')
    content = data.get('content')

    if not receiver_id or not content:
        return jsonify({"ошибка": "Все поля обязательны!"}), 400

    message = Message(sender_id=sender_id, receiver_id=receiver_id, content=content)
    try:
        db.session.add(message)
        db.session.commit()
        return jsonify({"сообщение": "Сообщение отправлено!"}), 201
    except Exception as e:
        return jsonify({"ошибка": "Ошибка при отправке сообщения."}), 400

@app.route('/messages/<int:user_id>', methods=['GET'])
@jwt_required()
def get_messages(user_id):
    current_user_id = get_jwt_identity()['id']
    messages = Message.query.filter(
        ((Message.sender_id == current_user_id) & (Message.receiver_id == user_id)) |
        ((Message.sender_id == user_id) & (Message.receiver_id == current_user_id))
    ).order_by(Message.timestamp).all()

    messages_list = [{
        "id": message.id,
        "sender_id": message.sender_id,
        "receiver_id": message.receiver_id,
        "content": message.content,
        "timestamp": message.timestamp
    } for message in messages]
    return jsonify(messages_list), 200

if __name__ == '__main__':
    app.run(debug=True)
