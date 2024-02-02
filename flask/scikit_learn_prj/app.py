from flask import (
    Flask, render_template, request, flash, redirect, url_for
    )
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename

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
    all_data = Data.query.all()
    return render_template('index.html', header=header, all_data=all_data)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        name = request.form['name']
        title = request.form['title']
        file = request.files['file']
        file_path = 'static/' + secure_filename(file.filename)
        file.save(file_path)
        register_data = Data(name=name, title=title, file_path=file_path)
        db.session.add(register_data)
        db.session.commit()
        flash('アップロードに成功しました')
        return redirect(url_for('index'))
    else:
        header = 'ファイルアップロード'
        return render_template('upload.html', header=header)

if __name__ == '__main__':
    app.run(debug=True)