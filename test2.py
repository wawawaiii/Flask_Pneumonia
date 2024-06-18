import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import exceptions as firebase_exceptions
from dotenv import load_dotenv
import logging
from datetime import timedelta

# 환경 변수 로드
load_dotenv()
cred_path = os.getenv('CRED_PATH')
secret_key = os.getenv('SECRET_KEY')

# Firebase 설정
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

# Flask 앱 설정
app = Flask(__name__)
app.secret_key = secret_key
app.permanent_session_lifetime = timedelta(minutes=30)
CORS(app)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/index', methods=['GET'])
def index_page():
    return render_template('index.html', user_logged_in='user_id' in session)

@app.route('/verifyToken', methods=['POST'])
def verify_token():
    id_token = request.json.get('idToken')
    logging.debug(f"Received ID Token: {id_token}")
    try:
        decoded_token = auth.verify_id_token(id_token, check_revoked=True)
        session['user_id'] = decoded_token['uid']
        return jsonify({'status': 'success', 'uid': decoded_token['uid']}), 200
    except ValueError as e:
        logging.error(f'Invalid token: {e}')
        return jsonify({'error': 'Invalid token: ' + str(e)}), 401
    except firebase_exceptions.FirebaseError as e:
        logging.error(f'Firebase error: {e}')
        return jsonify({'error': 'Firebase error: ' + str(e)}), 401
    except Exception as e:
        logging.error(f'General error: {e}')
        return jsonify({'error': 'General error: ' + str(e)}), 500

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_post():
    email = request.form['email']
    password = request.form['password']
    try:
        user = auth.create_user(email=email, password=password)
        session['user_id'] = user.uid
        return redirect(url_for('main'))  # 메인 페이지로 리다이렉트
    except Exception as e:
        logging.error(f'Registration error: {e}')
        return render_template('register.html', error=str(e))

@app.route('/main', methods=['GET'])
def main():
    return render_template('main.html', user_logged_in='user_id' in session)

@app.route('/', methods=['GET'])
def index():
    return render_template('main.html', user_logged_in='user_id' in session)

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
