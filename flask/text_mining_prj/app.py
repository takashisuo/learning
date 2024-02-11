from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


app = Flask(__name__)


# DB定義
URI = 'sqlite:///test.db'
app.config['SQLALCHEMY_DATABASE_URI'] = URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class DB(db.Model):
    __tablename__ = 'test_table'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(30), unique=True)
    file_path = db.Column(db.String(64))
    date = db.Column(db.DateTime, nullable=False, default=datetime.today())

# DB生成
@app.cli.command('initialize_DB')
def initialize_DB():
    db.create_all()

# エラーハンドリング
@app.errorhandler(404)
def not_found(error):
    return 'リンク先のページがないのでエラーです'

# index画面へのルーティング
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')




@app.route('/upload_register', methods=['POST'])
def upload_register():
    title = request.form['title']
    if title:
        file = request.files['file']
        file_path = secure_filename(file.filename)
        file.save(file_path)
        return file_path
    



@app.route('/paste')
def paste():
    return render_template('paste.html')

# 貼付されたデータを取得してテキストマイニングを実行する
@app.route('/paste_register', methods=['POST'])
def paste_register():
    title = request.form['title']
    if title:
        paste_data = request.form['paste_data']
        return paste_data



if __name__ == '__main__':
    app.run(debug=True)