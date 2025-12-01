from flask_login import UserMixin
from config import Config

import uuid


class Athlete(UserMixin, Config.db.Model):
    __tablename__ = 'athletes'
    id = Config.db.Column(Config.db.String(32), primary_key=True, default=str(uuid.uuid4()))
    nome = Config.db.Column(Config.db.String(50), nullable=False)
    data_nascimento = Config.db.Column(Config.db.Date, nullable=False)
    sexo = Config.db.Column(Config.db.String(10), nullable=False)
    estatura = Config.db.Column(Config.db.Float, nullable=False)
    envergadura = Config.db.Column(Config.db.Float, nullable=False)
    arremesso = Config.db.Column(Config.db.Float, nullable=False)
    salto_horizontal = Config.db.Column(Config.db.Float, nullable=False)
    abdominais = Config.db.Column(Config.db.Float, nullable=False)
    cluster = Config.db.Column(Config.db.Integer, nullable=False)
    
    
    def __init__(self, nome, data_nascimento, sexo, estatura, envergadura, arremesso, salto_horizontal, abdominais) -> None:
        self.nome = nome
        self.data_nascimento = data_nascimento
        self.sexo = sexo
        self.estatura = estatura
        self.envergadura = envergadura
        self.arremesso = arremesso
        self.salto_horizontal = salto_horizontal
        self.abdominais = abdominais
        self.cluster = -1  # Inicializa o cluster como -1 (não classificado)
        
    def dict(self) -> dict:
        return {
            'id': self.id,
            'nome': self.nome,
            'data_nascimento': self.data_nascimento,
            'sexo': self.sexo,
            'estatura': self.estatura,
            'envergadura': self.envergadura,
            'arremesso': self.arremesso,
            'salto_horizontal': self.salto_horizontal,
            'abdominais': self.abdominais,
            'cluster': self.cluster
        }
        