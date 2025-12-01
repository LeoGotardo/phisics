from flask_sqlalchemy import SQLAlchemy
from dataclasses import dataclass
from flask import Flask
from src.utils.dataclasses import Column

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
    
    nomeColumn = Column(name='nome', type='string', required=True, minLength=3, maxLength=50, description='Nome do atleta')
    dataNascimentoColumn = Column(name='dataNascimento', type='date', required=True, description='Data de nascimento do atleta')
    sexoColumn = Column(name='sexo', type='sex', required=True, description='Sexo do atleta')
    alturaColumn = Column(name='altura', type='num', required=True, minValue=0, maxValue=300, description='Altura do atleta')
    envergaduraColumn = Column(name='envergadura', type='num', required=True, minValue=0, maxValue=300, description='Envirgadura do atleta')
    arremessoColumn = Column(name='arremesso', type='num', required=True, minValue=0, maxValue=300, description='Arremesso do atleta')
    saltoHorizontalColumn = Column(name='saltoHorizontal', type='num', required=True, minValue=0, maxValue=300, description='Salto horizontal do atleta')
    abdominaisColumn = Column(name='abdominais', type='num', required=True, minValue=0, maxValue=300, description='Abdominais do atleta')
    
    COLUMNS = [nomeColumn, dataNascimentoColumn, sexoColumn, alturaColumn, envergaduraColumn, arremessoColumn, saltoHorizontalColumn, abdominaisColumn]