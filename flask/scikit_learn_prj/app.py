from flask import Flask
import os

app = Flask(__name__)
key = os.urandom(13)
app.secret_key = key

@app.route('/')
def index():
    return 'Hello Flask!!'

if __name__ == '__main__':
    app.run(debug=True)