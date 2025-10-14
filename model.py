from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from dotenv import load_dotenv
from flask import Flask

import locale, sys, os, uuid


class Config:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    load_dotenv()
    SECRET_KEY = os.getenv('SecretKey') 
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SECRET_KEY'] = SECRET_KEY
    db = SQLAlchemy(app)
    session = db.session
    
    
class Athlete(UserMixin, Config.db.Model):
    __tablename__ = 'athletes'
    id = Config.db.Column(Config.db.String(32), primary_key=True, default=str(uuid.uuid4()))
    nome = Config.db.Column(Config.db.String(50), nullable=False)
    data_nascimento = Config.db.Column(Config.db.Date, nullable=False)
    sexo = Config.db.Column(Config.db.String(10), nullable=False)
    massa_corporal = Config.db.Column(Config.db.Float, nullable=False)
    estatura = Config.db.Column(Config.db.Float, nullable=False)
    envergadura = Config.db.Column(Config.db.Float, nullable=False)
    arremesso = Config.db.Column(Config.db.Float, nullable=False)
    salto_horizontal = Config.db.Column(Config.db.Float, nullable=False)
    abdominais = Config.db.Column(Config.db.Float, nullable=False)
    
    
    def __init__(self, nome, data_nascimento, sexo, massa_corporal, estatura, envergadura, arremesso, salto_horizontal, abdominais):
        self.nome = nome
        self.data_nascimento = data_nascimento
        self.sexo = sexo
        self.massa_corporal = massa_corporal
        self.estatura = estatura
        self.envergadura = envergadura
        self.arremesso = arremesso
        self.salto_horizontal = salto_horizontal
        self.abdominais = abdominais
        
    def dict(self):
        return {
            'id': self.id,
            'nome': self.nome,
            'data_nascimento': self.data_nascimento,
            'sexo': self.sexo,
            'massa_corporal': self.massa_corporal,
            'estatura': self.estatura,
            'envergadura': self.envergadura,
            'arremesso': self.arremesso,
            'salto_horizontal': self.salto_horizontal,
            'abdominais': self.abdominais
        }
        
        
class Model:
    def __init__(self):
        self.db = Config.db
        self.session = Config.session
        
        self.create_tables()
        
        
    def create_tables(self):
        with Config.app.app_context():
            self.db.create_all()
        
        
    def getAthlete(self, id: str) -> tuple[bool, dict]:
        with Config.app.app_context():
            athlete = self.session.query(Athlete).filter(Athlete.id == id).first()
            return True, athlete.dict()
    
    
    def createAthlete(self, dados: list[dict]) -> tuple[bool, dict]:
        with Config.app.app_context():
            try:
                athlete = Athlete(**dados)
                self.session.add(athlete)
                self.session.commit()
                return True, athlete.dict()
            except Exception as e:
                self.session.rollback()
                raise e
    
    
    def getAthletes(self,  sort: str = 'id', sortOrder: str = 'desc', query: str = None, paginated: bool = False, page: int = 1, per_page: int = 10) -> tuple[bool, list[Athlete]]:
        
        sortOptions = {
            'name': Athlete.nome,
            'data': Athlete.data_nascimento,
            'sex': Athlete.sexo,
            'massa': Athlete.massa_corporal,
            'estatura': Athlete.estatura,
            'envergadura': Athlete.envergadura,
            'arremesso': Athlete.arremesso,
            'salto': Athlete.salto_horizontal,
            'abdominais': Athlete.abdominais
        }
        
        sortColumn = sortOptions.get(sort, Athlete.nome)
        
        with Config.app.app_context():
            baseQuery = self.session.query(Athlete)
            
        if query:
            baseQuery = baseQuery.filter(Athlete.nome.like(f'%{query}%'))
            
        if sortOrder == 'desc':
            baseQuery = baseQuery.order_by(sortColumn.desc())
        else:
            baseQuery = baseQuery.order_by(sortColumn.asc())
            
        if paginated:
            paginatedResults = baseQuery.paginate(page=page, per_page=per_page, error_out=False)
            
            athlets = [athlets.dict() for athletes in paginatedResults.items]
            
            currentPage = paginatedResults.page
            totalPages = paginatedResults.pages
            totalItems = paginatedResults.total
            
            startPage = max(1, currentPage - 5)
            endPage = min(totalPages, currentPage + 5)
            visiblePages = list(range(startPage, endPage + 1))
            
            athlets = {
                'items': athlets,
                'pagination': {
                    'currentPage': currentPage,
                    'totalPages': totalPages,
                        'total': paginatedResults.total,
                    'perPage': paginatedResults.per_page,
                    'hasPrev': paginatedResults.has_prev,
                    'hasNext': paginatedResults.has_next,
                    'prevPage': currentPage - 1 if paginatedResults.has_prev else None,
                    'nextPage': currentPage + 1 if paginatedResults.has_next else None,
                    'visiblePages': visiblePages,
                    'showFirst': 1 not in visiblePages,
                    'showLast': totalPages not in visiblePages,
                    'showLeftEllipsis': startPage > 2,
                    'showRightEllipsis': endPage < totalPages - 1
                },
                'filters': {
                    'query': query,
                    'sort': sort,
                    'sortOrder': sortOrder,
                }
            }
            
        else:
            athlets = baseQuery.all()
            athlets = [athlete.dict() for athlete in athlets]
            
        return True, athlets
            
    
    
    def deleteAthlete(self, id: str) -> tuple[bool, Athlete]:
        with Config.app.app_context():
            try:
                athlete = self.getAthlete(id)
                self.session.delete(athlete)
                self.session.commit()
                
                return True, "Atlete"
            except Exception as e:
                self.session.rollback()
                raise e