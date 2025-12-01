import  sys, sys

from elos.csvImportElo import CSVImportElo
from elos.csvExportElo import CSVExportElo
from src.model.elos.eloManager import EloManager
from athleteModel import Athlete
from knnModel import KNNModel
from sqlalchemy import func
from config import Config

        
class Model:
    def __init__(self):
        self.setupKMeans(n_clusters=3)
        self.CLUSTERS = ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']
        self.db = Config.db
        self.session = Config.session
        
        self.create_tables()
        
            
    def create_tables(self) -> None:
        with Config.app.app_context():
            self.db.create_all()
  
  
    def importCSV(self, file: bytes):
        elo = EloManager(CSVImportElo())
        
        try:
            athleteList = elo.startElo(file)
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
        return athleteList
    
    
    def exportCSV(self, fullData: bool = False, athletesID: list[str] = None, allAthletes: bool = False) -> tuple[bool, bytes] | tuple[bool, dict]:
        elo = EloManager(CSVExportElo())
        
        try:
            if not allAthletes and athletesID != None:
                csvFile = elo.startElo(athetesID=athletesID, fullData=fullData)
            else:
                csvFile = elo.startElo(fullData=fullData)
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
        return True, csvFile
    
    
    def classifyAthletes(self, athletes: list[Athlete | dict]) -> tuple[bool, list[dict]]:
        try:
            for athlete in athletes:
                if isinstance(athlete, dict):
                    athlete = Athlete(**athlete)
                    
                athlete.cluster = self.KNNModel.predict(athlete)
            
            success = self.createAthletes(athletes, rowData=False)
        except Exception as e:            
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
        return True, 
    
        
    
    def getDashboardInfo(self):
        try:
            totalAthletes = self.session.query(Athlete).count()
            elitAthletes = self.session.query(Athlete).filter(Athlete.cluster == 'Elite').count()
            competeAthletes = self.session.query(Athlete).filter(Athlete.cluster == 'Competitivo').count()
            intermedAthletes = self.session.query(Athlete).filter(Athlete.cluster == 'Intermediário').count()
            beginnerAthletes = self.session.query(Athlete).filter(Athlete.cluster == 'Iniciante').count()
            
            totalInfo = {'totalAthletes': totalAthletes}
            
            eliteCluster = {'label': 'Elite',
                            'count': elitAthletes,
                            "heightMed": self.session.query(Athlete).filter(Athlete.cluster == 'Elite').with_entities(func.avg(Athlete.estatura)).scalar(),
                            "sizeMed": self.session.query(Athlete).filter(Athlete.cluster == 'Elite').with_entities(func.avg(Athlete.envergadura)).scalar(),
                            "arremessoMed": self.session.query(Athlete).filter(Athlete.cluster == 'Elite').with_entities(func.avg(Athlete.arremesso)).scalar(),
                            "saltoMed": self.session.query(Athlete).filter(Athlete.cluster == 'Elite').with_entities(func.avg(Athlete.salto_horizontal)).scalar(),
                            "abdominMed": self.session.query(Athlete).filter(Athlete.cluster == 'Elite').with_entities(func.avg(Athlete.abdominais)).scalar()
                            }
            
            competitiveCluster = {'label': 'Competitivo',
                                'count': competeAthletes,
                                "heightMed": self.session.query(Athlete).filter(Athlete.cluster == 'Competitivo').with_entities(func.avg(Athlete.estatura)).scalar(),
                                "sizeMed": self.session.query(Athlete).filter(Athlete.cluster == 'Competitivo').with_entities(func.avg(Athlete.envergadura)).scalar(),
                                "arremessoMed": self.session.query(Athlete).filter(Athlete.cluster == 'Competitivo').with_entities(func.avg(Athlete.arremesso)).scalar(),
                                "saltoMed": self.session.query(Athlete).filter(Athlete.cluster == 'Competitivo').with_entities(func.avg(Athlete.salto_horizontal)).scalar(),
                                "abdominMed": self.session.query(Athlete).filter(Athlete.cluster == 'Competitivo').with_entities(func.avg(Athlete.abdominais)).scalar()
                                }
            
            intemedianCluster = {'label': 'Intermediário',
                                'count': intermedAthletes,
                                "heightMed": self.session.query(Athlete).filter(Athlete.cluster == 'Intermediario').with_entities(func.avg(Athlete.estatura)).scalar(),
                                "sizeMed": self.session.query(Athlete).filter(Athlete.cluster == 'Intermediario').with_entities(func.avg(Athlete.envergadura)).scalar(),
                                "arremessoMed": self.session.query(Athlete).filter(Athlete.cluster == 'Intermediario').with_entities(func.avg(Athlete.arremesso)).scalar(),
                                "saltoMed": self.session.query(Athlete).filter(Athlete.cluster == 'Intermediario').with_entities(func.avg(Athlete.salto_horizontal)).scalar(),
                                "abdominMed": self.session.query(Athlete).filter(Athlete.cluster == 'Intermediario').with_entities(func.avg(Athlete.abdominais)).scalar()
                                }
            
            beginnerCluster = {'label': 'Iniciante',
                                'count': beginnerAthletes,
                                "heightMed": self.session.query(Athlete).filter(Athlete.cluster == 'Iniciante').with_entities(func.avg(Athlete.estatura)).scalar(),
                                "sizeMed": self.session.query(Athlete).filter(Athlete.cluster == 'Iniciante').with_entities(func.avg(Athlete.envergadura)).scalar(),
                                "arremessoMed": self.session.query(Athlete).filter(Athlete.cluster == 'Iniciante').with_entities(func.avg(Athlete.arremesso)).scalar(),
                                "saltoMed": self.session.query(Athlete).filter(Athlete.cluster == 'Iniciante').with_entities(func.avg(Athlete.salto_horizontal)).scalar(),
                                "abdominMed": self.session.query(Athlete).filter(Athlete.cluster == 'Iniciante').with_entities(func.avg(Athlete.abdominais)).scalar()
                                }

            

            info = {
                'totalInfo': totalInfo,
                'clusterInfo': {
                    'eliteCluster': eliteCluster,
                    'competitiveCluster': competitiveCluster,
                    'intermediateCluster': intemedianCluster,
                    'beginerCluster': beginnerCluster
            }
            }
            
            return True, info
                
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
    
    
    def getAthletes(self, sort: str = 'id', sortOrder: str = 'desc', query: str = None, paginated: bool = False, page: int = 1, per_page: int = 10) -> tuple[bool, list[Athlete]] | tuple[bool, dict]:
        
        sortOptions = {
            'name': Athlete.nome,
            'data': Athlete.data_nascimento,
            'sex': Athlete.sexo,
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
            
            athlets = [athlete.dict() for athlete in paginatedResults.items]
            
            currentPage = paginatedResults.page
            totalPages = paginatedResults.pages
            
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
    
    
    def createAthletes(self, athletes: list[dict] | list[Athlete], rowData: bool = False) -> bool:
        try:
            for athlete in athletes:
                if rowData:
                    athlete = Athlete(**athlete)
                self.session.add(athlete)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        return True