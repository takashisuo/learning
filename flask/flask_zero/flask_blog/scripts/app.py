import click
from flask import Flask
from flask.cli import with_appcontext
#from flask_blog import db
from flask_blog import app
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

#app = Flask(__name__)
#app.config.from_object('flask_blog.config')

db = SQLAlchemy(app)

class Entry(db.Model):
    __tablename__ = 'entries'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50), unique=True)
    text = db.Column(db.Text)
    created_at = db.Column(db.DateTime)

    def __init__(self, title=None, text=None):
        self.title = title
        self.text = text
        self.created_at = datetime.utcnow()

    def __repr__(self):
        return '<Entry id:{} title:{} text{}>'.format(self.id, self.title, self.text)

@app.cli.command('init_db')
def init_db_command():
    db.create_all()
    click.echo('initialize')
