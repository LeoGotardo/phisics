import sys, pandas as pd

from src.model.elos.dataAnalysisElo import DataAnalysisElo
from src.model.elos.dbAnalyticsElo import DBAnalyticsElo
from src.model.elos.csvImportElo import CSVImportElo
from src.model.elos.viewDataElo import ViewDataElo
from src.utils.dataGenerator import DataGenerator
from src.model.elos.eloManager import EloManager
from typing import Tuple, List, Dict, Literal
from src.model.athleteModel import Athlete
from src.model.knnModel import KNNModel
from src.config import Config
from sqlalchemy import func
from icecream import ic

        
class Model:
    def __init__(self):
        self.db = Config.db
        self.session = Config.session
        self.knnModel = KNNModel(nNeighbors=5)
        self.create_tables()
        
        # Tentar carregar modelo salvo
        status, msg = self.knnModel.loadModel('models/knn_model.pkl')
        if status:
            print(f"✓ {msg}")
        else:
            print(f"⚠ Modelo não carregado, será necessário treinar: {msg}")
            data = DataGenerator().generateData()
            success, msg = self.knnModel.fit(data)
            if success != True:
                print(f"⚠ Não foi possível treinar o modelo: {msg}")

        
        
            
    def create_tables(self) -> None:
        with Config.app.app_context():
            self.db.create_all()
  
  
    def loadCSVData(self, file: bytes) -> Tuple[Literal[True, False, -1], str]:
        """
        Carrega dados de arquivo CSV, valida e insere no banco.
        
        Args:
            file: Bytes do arquivo CSV
            
        Returns:
            Tupla (status, mensagem)
        """
        try:
            # Criar manager do elo de importação
            elo = EloManager(CSVImportElo())
            
            # Processar CSV através da cadeia
            athletesList = elo.startElo(file)
            
            ic(athletesList)
            
            if not athletesList or len(athletesList) == 0:
                return False, 'Nenhum atleta válido encontrado no CSV'
            
            for athlete in athletesList:
                success, result = self.putAthlete(athlete)
                if success != True:
                    return success, result
            
            return True, f'{len(athletesList)} atletas importados com sucesso!'
            
        except ValueError as e:
            # Erros de validação
            return False, str(e)
            
        except Exception as e:
            # Rollback em caso de erro
            self.session.rollback()
            
            error_msg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno} '
                        f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, error_msg
 
 
    def clearDatabase(self) -> Tuple[bool, str]:
        """
        Limpa o banco de dados.
        
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            with Config.app.app_context():
                self.session.query(Athlete).delete()
                self.session.commit()
                
                return True, 'Banco de dados limpo com sucesso!'
                
        except Exception as e:
            self.session.rollback()
            error_msg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno} '
                        f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, error_msg
 
 
    def exportData(self) -> Tuple[bool, bytes | str]:
        """
        Exporta dados do banco para CSV.
            
        Returns:
            Tupla (sucesso, bytes_do_arquivo_ou_mensagem_de_erro)
        """
        try:
            with Config.app.app_context():
                athletes = self.session.query(Athlete).all()
                
                if not athletes or len(athletes) == 0:
                    return False, 'Nenhum atleta encontrado no banco de dados'
                
                # Converter para DataFrame
                athletesData = [athlete.dict() for athlete in athletes]
                dfAthletes = pd.DataFrame(athletesData)
                
                essentialColumns = ['id', 'nome', 'dataNascimento', 'sexo', 'altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais', 'cluster']
                dfAthletes = dfAthletes[essentialColumns]
                
                # Converter DataFrame para CSV em bytes
                csvBytes = dfAthletes.to_csv(index=False).encode('utf-8')
                
                return True, csvBytes
                
        except Exception as e:
            error_msg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno} '
                        f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, error_msg
    
 
    def trainKNNModel(self, forceRetrain: bool = False) -> Tuple[bool, str]:
        """
        Treina o modelo KNN com todos os atletas do banco.
        
        Args:
            forceRetrain: Se True, retreina mesmo se já estiver treinado
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            if self.knnModel.isTrained and not forceRetrain:
                return True, "Modelo já está treinado. Use forceRetrain=True para retreinar."
            
            # Buscar todos os atletas do banco
            athletes = self.session.query(Athlete).all()
            
            if len(athletes) < 20:
                return False, f"Necessário pelo menos 20 atletas para treinar. Encontrados: {len(athletes)}"
            
            # Converter para DataFrame
            athletesData = [athlete.dict() for athlete in athletes]
            dfAthletes = pd.DataFrame(athletesData)
            
            # Treinar modelo
            status, msg = self.knnModel.fit(dfAthletes)
            
            if status:
                # Salvar modelo
                self.knnModel.saveModel('models/knn_model.pkl')
                return True, msg
            else:
                return False, msg
                
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
        
        
    def getAthleteById(self, athleteId: str) -> Tuple[bool, Athlete | None]:
        """
        Busca um atleta pelo seu ID.
        
        Args:
            athleteId: ID do atleta
            
        Returns:
            Tupla (status, atleta_ou_None)
        """
        try:
            with Config.app.app_context():
                athlete = self.session.query(Athlete).filter_by(id=athleteId).first()
                return True, athlete
                
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def classifyAthlete(self, athleteData: Dict | Athlete) -> Tuple[bool, Dict | str]:
        """
        Classifica um atleta usando o modelo KNN.
        
        Args:
            athleteData: Dados do atleta (dict ou objeto Athlete)
            
        Returns:
            Tupla (sucesso, resultado)
        """
        try:
            # Verificar se modelo está treinado
            if not self.knnModel.isTrained:
                return False, "Modelo KNN não foi treinado. Execute trainKNNModel() primeiro."
            
            # Converter Athlete para dict se necessário
            if isinstance(athleteData, Athlete):
                athleteData = athleteData.dict()
            
            # Fazer predição
            status, resultado = self.knnModel.predict(athleteData)
            
            return status, resultado
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def classifyAthletes(self, athletesList: List[Athlete | Dict]) -> Tuple[bool, List[Dict] | str]:
        """
        Classifica múltiplos atletas e atualiza no banco.
        
        Args:
            athletesList: Lista de atletas para classificar
            
        Returns:
            Tupla (sucesso, resultados)
        """
        try:
            if not self.knnModel.isTrained:
                # Tentar treinar automaticamente
                trainStatus, trainMsg = self.trainKNNModel()
                if not trainStatus:
                    return False, f"Não foi possível treinar o modelo: {trainMsg}"
            
            results = []
            
            for athlete in athletesList:
                # Converter para dict se necessário
                if isinstance(athlete, Athlete):
                    athleteDict = athlete.dict()
                    athleteObj = athlete
                else:
                    athleteDict = athlete
                    athleteObj = Athlete(**athlete)
                
                # Classificar
                status, resultado = self.knnModel.predict(athleteDict)
                
                if status:
                    # Atualizar cluster do atleta
                    athleteObj.cluster = resultado['cluster']
                    
                    # Adicionar ao banco se ainda não estiver
                    existing = self.session.query(Athlete).filter_by(id=athleteObj.id).first()
                    if not existing:
                        self.session.add(athleteObj)
                    else:
                        existing.cluster = resultado['cluster']
                    
                    results.append({
                        'nome': athleteObj.nome,
                        'cluster': resultado['cluster'],
                        'confianca': resultado['confianca']
                    })
            
            # Commit das alterações
            self.session.commit()
            
            return True, results
            
        except Exception as e:
            self.session.rollback()
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def findSimilarAthletes(self, athleteId: str, nSimilar: int = 5) -> Tuple[bool, List[Dict] | str]:
        """
        Encontra atletas similares a um atleta específico.
        
        Args:
            athleteId: ID do atleta de referência
            nSimilar: Número de atletas similares a retornar
            
        Returns:
            Tupla (sucesso, lista de atletas similares)
        """
        try:
            # Buscar atleta
            athlete = self.session.query(Athlete).filter_by(id=athleteId).first()
            
            if not athlete:
                return False, f"Atleta com ID {athleteId} não encontrado"
            
            # Buscar similares
            status, similares = self.knnModel.findSimilarAthletes(
                athlete.dict(),
                nSimilar=nSimilar
            )
            
            if not status:
                return False, similares
            
            # Buscar dados completos dos atletas similares
            allAthletes = self.session.query(Athlete).all()
            
            similaresCompletos = []
            for similar in similares:
                idx = similar['indice']
                if idx < len(allAthletes):
                    athleteSimilar = allAthletes[idx]
                    similaresCompletos.append({
                        'nome': athleteSimilar.nome,
                        'cluster': athleteSimilar.cluster,
                        'similaridade': similar['similaridade'],
                        'distancia': similar['distancia'],
                        'dados': athleteSimilar.dict()
                    })
            
            return True, similaresCompletos
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def getKNNMetrics(self) -> Tuple[bool, Dict | str]:
        """
        Retorna métricas de qualidade do modelo KNN.
        
        Returns:
            Tupla (sucesso, métricas)
        """
        try:
            if not self.knnModel.isTrained:
                return False, "Modelo não treinado"
            
            # Buscar atletas para teste
            athletes = self.session.query(Athlete).all()
            athletesData = [athlete.dict() for athlete in athletes]
            dfAthletes = pd.DataFrame(athletesData)
            
            # Obter métricas
            metrics = self.knnModel.getModelMetrics(dfAthletes)
            
            return True, metrics
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def optimizeKNN(self) -> Tuple[bool, Dict | str]:
        """
        Otimiza o valor de K do modelo KNN.
        
        Returns:
            Tupla (sucesso, resultados da otimização)
        """
        try:
            # Buscar atletas
            athletes = self.session.query(Athlete).all()
            
            if len(athletes) < 20:
                return False, "Necessário pelo menos 20 atletas"
            
            athletesData = [athlete.dict() for athlete in athletes]
            dfAthletes = pd.DataFrame(athletesData)
            
            # Otimizar K
            bestK, results = self.knnModel.optimizeKValue(dfAthletes)
            
            # Retreinar com melhor K
            trainStatus, trainMsg = self.knnModel.fit(dfAthletes)
            
            if trainStatus:
                # Salvar modelo otimizado
                self.knnModel.saveModel('models/knn_model.pkl')
                
                return True, {
                    'melhorK': bestK,
                    'resultados': results,
                    'mensagem': f"Modelo otimizado com K={bestK} e retreinado"
                }
            else:
                return False, trainMsg
                
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def getFeatureImportance(self) -> Tuple[bool, Dict | str]:
        """
        Retorna a importância de cada feature na classificação.
        
        Returns:
            Tupla (sucesso, importância das features)
        """
        try:
            athletes = self.session.query(Athlete).all()
            athletesData = [athlete.dict() for athlete in athletes]
            dfAthletes = pd.DataFrame(athletesData)
            
            importance = self.knnModel.getFeatureImportance(dfAthletes)
            
            return True, importance
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def getDashboardInfo(self):
        """
        Obtém informações agregadas para o dashboard.
        Corrige queries para usar IDs numéricos dos clusters.
        """
        try:
            with Config.app.app_context():
                # Total de atletas
                totalAthletes = self.session.query(Athlete).count()
                
                # Contagem por cluster (usando IDs: 0=Iniciante, 1=Intermediário, 2=Competitivo, 3=Elite)
                elitAthletes = self.session.query(Athlete).filter(Athlete.cluster == 3).count()
                competeAthletes = self.session.query(Athlete).filter(Athlete.cluster == 2).count()
                intermedAthletes = self.session.query(Athlete).filter(Athlete.cluster == 1).count()
                beginnerAthletes = self.session.query(Athlete).filter(Athlete.cluster == 0).count()
                
                totalInfo = {'totalAthletes': totalAthletes}
                
                # Elite (cluster 3)
                eliteCluster = {
                    'label': 'Elite',
                    'count': elitAthletes,
                    "heightMed": round(self.session.query(func.avg(Athlete.altura))
                        .filter(Athlete.cluster == 3).scalar() or 0, 2),
                    "sizeMed": round(self.session.query(func.avg(Athlete.envergadura))
                        .filter(Athlete.cluster == 3).scalar() or 0, 2),
                    "arremessoMed": round(self.session.query(func.avg(Athlete.arremesso))
                        .filter(Athlete.cluster == 3).scalar() or 0, 2),
                    "saltoMed": round(self.session.query(func.avg(Athlete.saltoHorizontal))
                        .filter(Athlete.cluster == 3).scalar() or 0, 2),
                    "abdominMed": round(self.session.query(func.avg(Athlete.abdominais))
                        .filter(Athlete.cluster == 3).scalar() or 0, 2)
                }
                
                # Competitivo (cluster 2)
                competitiveCluster = {
                    'label': 'Competitivo',
                    'count': competeAthletes,
                    "heightMed": round(self.session.query(func.avg(Athlete.altura))
                        .filter(Athlete.cluster == 2).scalar() or 0, 2),
                    "sizeMed": round(self.session.query(func.avg(Athlete.envergadura))
                        .filter(Athlete.cluster == 2).scalar() or 0, 2),
                    "arremessoMed": round(self.session.query(func.avg(Athlete.arremesso))
                        .filter(Athlete.cluster == 2).scalar() or 0, 2),
                    "saltoMed": round(self.session.query(func.avg(Athlete.saltoHorizontal))
                        .filter(Athlete.cluster == 2).scalar() or 0, 2),
                    "abdominMed": round(self.session.query(func.avg(Athlete.abdominais))
                        .filter(Athlete.cluster == 2).scalar() or 0, 2)
                }
                
                # Intermediário (cluster 1)
                intermediateCluster = {
                    'label': 'Intermediário',
                    'count': intermedAthletes,
                    "heightMed": round(self.session.query(func.avg(Athlete.altura))
                        .filter(Athlete.cluster == 1).scalar() or 0, 2),
                    "sizeMed": round(self.session.query(func.avg(Athlete.envergadura))
                        .filter(Athlete.cluster == 1).scalar() or 0, 2),
                    "arremessoMed": round(self.session.query(func.avg(Athlete.arremesso))
                        .filter(Athlete.cluster == 1).scalar() or 0, 2),
                    "saltoMed": round(self.session.query(func.avg(Athlete.saltoHorizontal))
                        .filter(Athlete.cluster == 1).scalar() or 0, 2),
                    "abdominMed": round(self.session.query(func.avg(Athlete.abdominais))
                        .filter(Athlete.cluster == 1).scalar() or 0, 2)
                }
                
                # Iniciante (cluster 0)
                beginnerCluster = {
                    'label': 'Iniciante',
                    'count': beginnerAthletes,
                    "heightMed": round(self.session.query(func.avg(Athlete.altura))
                        .filter(Athlete.cluster == 0).scalar() or 0, 2),
                    "sizeMed": round(self.session.query(func.avg(Athlete.envergadura))
                        .filter(Athlete.cluster == 0).scalar() or 0, 2),
                    "arremessoMed": round(self.session.query(func.avg(Athlete.arremesso))
                        .filter(Athlete.cluster == 0).scalar() or 0, 2),
                    "saltoMed": round(self.session.query(func.avg(Athlete.saltoHorizontal))
                        .filter(Athlete.cluster == 0).scalar() or 0, 2),
                    "abdominMed": round(self.session.query(func.avg(Athlete.abdominais))
                        .filter(Athlete.cluster == 0).scalar() or 0, 2)
                }

                info = {
                    'totalInfo': totalInfo,
                    'clusterInfo': {
                        'eliteCluster': eliteCluster,
                        'competitiveCluster': competitiveCluster,
                        'intermediateCluster': intermediateCluster,
                        'beginerCluster': beginnerCluster
                    }
                }
                
                return True, info
                
        except Exception as e:
            error_msg = f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
            return -1, error_msg
    
    
    def getAthletes(self, id: str = None, ageRange: tuple[int|int] = None, cluster: str = None, sort: str = 'id', sortOrder: str = 'desc', query: str = None, paginated: bool = False, page: int = 1, per_page: int = 10) -> tuple[bool, list[Athlete]] | tuple[bool, dict]:
        try:
            sortOptions = {
                'nome': Athlete.nome,
                'dataNascimento': Athlete.dataNascimento,
                'sexo': Athlete.sexo,
                'altura': Athlete.altura,
                'envergadura': Athlete.envergadura,
                'arremesso': Athlete.arremesso,
                'saltoHorizontal': Athlete.saltoHorizontal,
                'abdominais': Athlete.abdominais,
                'cluster': Athlete.cluster
            }
            
            sortColumn = sortOptions.get(sort, Athlete.nome)
            
            with Config.app.app_context():
                baseQuery = self.session.query(Athlete)
                
            # Filtros
            if query:
                baseQuery = baseQuery.filter(Athlete.nome.like(f'%{query}%'))
            
            if id:
                baseQuery = baseQuery.filter(Athlete.id == id)
            
            # CORREÇÃO: Filtro de idade usando dataNascimento
            if ageRange:
                data_min, data_max = ageRange
                baseQuery = baseQuery.filter(
                    Athlete.dataNascimento.between(data_min, data_max)
                )
            
            # CORREÇÃO: Mapear nomes de cluster para IDs
            if cluster:
                cluster_map = {
                    'elite': 3,
                    'competitivo': 2,
                    'intermediário': 1,
                    'intermediario': 1,  # Aceitar sem acento também
                    'iniciante': 0
                }
                cluster_id = cluster_map.get(cluster.lower())
                if cluster_id is not None:
                    baseQuery = baseQuery.filter(Athlete.cluster == cluster_id)
            
            # Ordenação
            if sortOrder == 'desc':
                baseQuery = baseQuery.order_by(sortColumn.desc())
            else:
                baseQuery = baseQuery.order_by(sortColumn.asc())
            
            if paginated:
                paginatedResults = baseQuery.paginate(page=int(page), per_page=int(per_page), error_out=False)
                
                athletes = [athlete.dict() for athlete in paginatedResults.items]
                
                currentPage = paginatedResults.page
                totalPages = paginatedResults.pages
                
                startPage = max(1, currentPage - 5)
                endPage = min(totalPages, currentPage + 5)
                visiblePages = list(range(startPage, endPage + 1))
                
                result = {
                    'items': athletes,
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
                        'query': query or '',
                        'sort': sort,
                        'sortOrder': sortOrder,
                        'cluster': cluster or 'all',
                        'ageRange': 'all'  # Será sobrescrito no controller
                    }
                }
                
            else:
                athletes = baseQuery.all()
                result = [athlete.dict() for athlete in athletes]
                
            return True, result
            
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
    
    
    def getViewInfo(self) -> Tuple[Literal[True, False, -1], Dict | str]:
        """
        Obtém todas as informações necessárias para a página de visualização.
        Usa o ViewDataElo para processar os dados através de uma cadeia.
        
        Returns:
            Tupla (status, dados)
            - True: sucesso, retorna dicionário com dados
            - False: erro leve, retorna mensagem
            - -1: erro grave, retorna mensagem de erro
        """
        try:
            elo = EloManager(ViewDataElo())
            view_data = elo.startElo()
            return True, view_data
            
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            error_msg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno} '
                        f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, error_msg
    
    
    def putAthlete(self, athleteData: dict | Athlete) -> Tuple[Literal[True, False, -1], str]:
        """
        Cria um novo atleta no banco de dados.
        
        Args:
            athleteData: Dicionário com dados do atleta
            
        Returns:
            Tupla (status, mensagem)
        """
        try:
            athlete = Athlete(**athleteData)

            success, clusterResult = self.knnModel.predict(athleteData)
            
            if success != True:
                return -1, clusterResult
            
            # CORREÇÃO: Extrair o valor numérico do cluster
            if isinstance(clusterResult, dict):
                clusterName = clusterResult['cluster']
                # Mapear nome do cluster para ID numérico
                clusterMapping = {
                    'Iniciante': 0,
                    'Intermediário': 1,
                    'Competitivo': 2,
                    'Elite': 3
                }
                athlete.cluster = clusterMapping.get(clusterName, 0)
            elif isinstance(clusterResult, (int, float)):
                athlete.cluster = int(clusterResult)
            elif isinstance(clusterResult, str):
                # Se for string, tentar mapear
                clusterMapping = {
                    'Iniciante': 0,
                    'Intermediário': 1,
                    'Competitivo': 2,
                    'Elite': 3
                }
                athlete.cluster = clusterMapping.get(clusterResult, 0)
            else:
                # Fallback
                athlete.cluster = 0
            
            self.session.add(athlete)
            self.session.commit()
            
            clusterNameDisplay = athlete.getClusterName()
            
            return True, f'Atleta {athlete.nome} cadastrado com sucesso no cluster {clusterNameDisplay}!'
            
        except Exception as e:
            self.session.rollback()
            errorMsg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno}')
            return -1, errorMsg
            
    
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
    
    
    def getDataAnalysis(self) -> Tuple[Literal[True, False, -1], Dict | str]:
        """
        Obtém análise estatística completa dos dados.
        
        Returns:
            Tupla (status, dados_ou_mensagem)
        """
        try:
            elo = EloManager(DataAnalysisElo())
            analysis = elo.startElo()
            
            return True, analysis
            
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            error_msg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno} '
                        f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, error_msg


    def getDBAnalytics(self) -> Tuple[Literal[True, False, -1], Dict | str]:
        """
        Obtém analytics do banco de dados.
        
        Returns:
            Tupla (status, dados_ou_mensagem)
        """
        try:
            elo = EloManager(DBAnalyticsElo())
            analytics = elo.startElo()
            
            return True, analytics
            
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            error_msg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno} '
                        f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, error_msg


    def getCompleteAnalysis(self) -> Tuple[Literal[True, False, -1], Dict | str]:
        """
        Obtém análise completa combinando ViewData, DataAnalysis e DBAnalytics.
        Ideal para geração de relatórios completos.
        
        Returns:
            Tupla (status, dados_completos_ou_mensagem)
        """
        try:
            # Obter dados de visualização (PCA, gráficos, etc)
            viewStatus, viewData = self.getViewInfo()
            
            if viewStatus != True:
                return viewStatus, viewData
            
            # Obter análise estatística
            analysisStatus, analysisData = self.getDataAnalysis()
            
            if analysisStatus != True:
                return analysisStatus, analysisData
            
            # Obter analytics do banco
            dbStatus, dbData = self.getDBAnalytics()
            
            if dbStatus != True:
                return dbStatus, dbData
            
            # Combinar todos os dados
            completeData = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'visualizacao': viewData,
                'analise_estatistica': analysisData,
                'analytics_banco': dbData,
                'resumo_executivo': {
                    'total_atletas': dbData['estatisticas_basicas']['total_atletas'],
                    'clusters': len(viewData['estatisticas']['distribuicao']),
                    'qualidade_clustering': viewData['metricas']['silhouette'],
                    'insights_principais': analysisData['insights'][:3],
                    'anomalias': len(dbData['analytics'].get('anomalias', {}).get('outliers', [])),
                    'top_performers': dbData['analytics']['rankings']['top_geral'][:5]
                }
            }
            
            return True, completeData
            
        except Exception as e:
            error_msg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno} '
                        f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, error_msg


    def updateAthlete(self, athleteId: str, athleteData: dict) -> Tuple[Literal[True, False, -1], str]:
        """
        Atualiza os dados de um atleta existente.
        
        Args:
            athleteId: ID do atleta
            athleteData: Dicionário com novos dados
            
        Returns:
            Tupla (status, mensagem)
        """
        try:
            with Config.app.app_context():
                athlete = self.session.query(Athlete).filter_by(id=athleteId).first()
                
                if not athlete:
                    return False, f'Atleta com ID {athleteId} não encontrado'
                
                # Atualizar campos
                if 'nome' in athleteData:
                    athlete.nome = athleteData['nome']
                if 'dataNascimento' in athleteData:
                    athlete.dataNascimento = athleteData['dataNascimento']
                if 'sexo' in athleteData:
                    athlete.sexo = athleteData['sexo']
                if 'altura' in athleteData:
                    athlete.altura = float(athleteData['altura'])
                if 'envergadura' in athleteData:
                    athlete.envergadura = float(athleteData['envergadura'])
                if 'arremesso' in athleteData:
                    athlete.arremesso = float(athleteData['arremesso'])
                if 'saltoHorizontal' in athleteData:
                    athlete.saltoHorizontal = float(athleteData['saltoHorizontal'])
                if 'abdominais' in athleteData:
                    athlete.abdominais = int(athleteData['abdominais'])
                
                # Reclassificar com KNN se alguma métrica mudou
                if any(k in athleteData for k in ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais', 'sexo']):
                    if self.knnModel.isTrained:
                        success, result = self.knnModel.predict(athlete.dict())
                        if success and isinstance(result, dict):
                            athlete.cluster = result['cluster']
                
                self.session.commit()
                
                return True, f'Atleta {athlete.nome} atualizado com sucesso!'
                
        except Exception as e:
            self.session.rollback()
            error_msg = f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
            return -1, error_msg


    def deleteAthlete(self, athleteId: str) -> Tuple[Literal[True, False, -1], str]:
        """
        Exclui um atleta do banco de dados.
        
        Args:
            athleteId: ID do atleta
            
        Returns:
            Tupla (status, mensagem)
        """
        try:
            with Config.app.app_context():
                athlete = self.session.query(Athlete).filter_by(id=athleteId).first()
                
                if not athlete:
                    return False, f'Atleta com ID {athleteId} não encontrado'
                
                nome = athlete.nome
                self.session.delete(athlete)
                self.session.commit()
                
                return True, f'Atleta {nome} excluído com sucesso!'
                
        except Exception as e:
            self.session.rollback()
            error_msg = f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
            return -1, error_msg