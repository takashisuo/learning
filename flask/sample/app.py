from flask import Flask
from flask import render_template
from flask import make_response
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    """画面表示
    Returns:
    render_template: index.html
    """
    return render_template('index.html')

@app.route('/hello', methods=["POST"])
def hello():
    name = request.form.get("name")
    return make_response(f"こんにちは, {name} さん")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)