from flask_login import UserMixin
from src.config import Config
from datetime import datetime

import uuid


class Athlete(UserMixin, Config.db.Model):
    __tablename__ = 'athletes'
    id = Config.db.Column(Config.db.String(32), primary_key=True, default=lambda: str(uuid.uuid4()))
    nome = Config.db.Column(Config.db.String(50), nullable=False)
    dataNascimento = Config.db.Column(Config.db.Date, nullable=False)
    sexo = Config.db.Column(Config.db.String(10), nullable=False)
    altura = Config.db.Column(Config.db.Float, nullable=False)
    envergadura = Config.db.Column(Config.db.Float, nullable=False)
    arremesso = Config.db.Column(Config.db.Float, nullable=False)
    saltoHorizontal = Config.db.Column(Config.db.Float, nullable=False)
    abdominais = Config.db.Column(Config.db.Float, nullable=False)
    cluster = Config.db.Column(Config.db.Integer, nullable=False)
    
    
    def __init__(self, nome, dataNascimento, sexo, altura, envergadura, arremesso, saltoHorizontal, abdominais) -> None:
        self.nome = nome
        self.dataNascimento = dataNascimento
        self.sexo = sexo
        self.altura = altura
        self.envergadura = envergadura
        self.arremesso = arremesso
        self.saltoHorizontal = saltoHorizontal
        self.abdominais = abdominais
        self.cluster = -1  # Inicializa o cluster como -1 (não classificado)
        
    def dict(self) -> dict:
        return {
            'id': self.id,
            'nome': self.nome,
            'dataNascimento': self.dataNascimento,
            'sexo': self.sexo,
            'altura': self.altura,
            'envergadura': self.envergadura,
            'arremesso': self.arremesso,
            'saltoHorizontal': self.saltoHorizontal,
            'abdominais': self.abdominais,
            'cluster': self.cluster,
            'idade': int(datetime.now().year - self.dataNascimento.year)
        }
        