from flask_sqlalchemy import SQLAlchemy
from dataclasses import dataclass
from flask import Flask

import locale


@dataclass
class Config:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    SECRET_KEY = "secret_key"
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SECRET_KEY'] = SECRET_KEY
    db = SQLAlchemy(app)
    session = db.session