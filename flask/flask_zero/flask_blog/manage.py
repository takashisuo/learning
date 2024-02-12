from flask.cli import AppGroup
from flask_blog import app
from flask.flask_zero.flask_blog.scripts.app import InitDB

if __name__ == "__main__":
    db_cli = AppGroup('init_db', help='database related commands')
    app.cli.add_command(db_cli)