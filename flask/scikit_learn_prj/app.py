from flask import Flask, render_template
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
key = os.urandom(13)
app.secret_key = key



URI = 'sqlite:///file.db'
app.config['SQLALCHEMY_DATABASE_URI'] = URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# モデルの準備
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30), unique=False)
    title = db.Column(db.String(30), unique=False)
    file_path = db.Column(db.String(64), index=True, unique=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.now())

# データベースの初期化コマンド
@app.cli.command('initdb')
def initialize_DB():
    db.create_all()

@app.route('/')
def index():
    header = 'クラスタリングアプリケーション'
    return render_template('index.html', header=header)

if __name__ == '__main__':
    app.run(debug=True)