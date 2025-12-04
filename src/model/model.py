import sys, pandas as pd


from src.model.elos.dataAnalysisElo import DataAnalysisElo
from src.model.elos.dbAnalyticsElo import DBAnalyticsElo
from src.model.elos.csvImportElo import CSVImportElo
from src.model.elos.csvExportElo import CSVExportElo
from src.model.elos.viewDataElo import ViewDataElo
from src.utils.dataGenerator import DataGenerator
from src.model.elos.eloManager import EloManager
from typing import Tuple, List, Dict, Literal
from src.model.athleteModel import Athlete
from src.model.knnModel import KNNModel
from datetime import datetime
from src.config import Config
from sqlalchemy import func
from icecream import ic

        
class Model:
    def __init__(self):
        self.db = Config.db
        self.session = Config.session
        self.knnModel = KNNModel(nNeighbors=5)
        
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

        
        self.create_tables()
        
            
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
            
            if not athletesList or len(athletesList) == 0:
                return False, 'Nenhum atleta válido encontrado no CSV'
            
            # Inserir atletas no banco
            with Config.app.app_context():
                for athlete in athletesList:
                    self.session.add(athlete)
                
                self.session.commit()
            
            # Verificar se precisa treinar/retreinar o modelo KNN
            total_athletes = self.session.query(Athlete).count()
            
            if total_athletes >= 20:
                # Retreinar modelo com novos dados
                train_status, train_msg = self.trainKNNModel(forceRetrain=True)
                
                if train_status:
                    return True, f'{len(athletesList)} atletas importados e modelo retreinado com sucesso!'
                else:
                    return True, f'{len(athletesList)} atletas importados (aviso: {train_msg})'
            
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
    
    
    def exportData(self, fullData: bool = True, athletesIds: list[str] = None) -> Tuple[Literal[True, False, -1], bytes | str]:
        """
        Exporta dados dos atletas em formato ZIP (CSV + gráficos) ou apenas CSV.
        
        Args:
            fullData: Se True, exporta ZIP completo. Se False, apenas CSV
            athletesIds: Lista de IDs específicos para exportar (None = todos)
            
        Returns:
            Tupla (status, dados_ou_mensagem)
        """
        try:
            # Criar manager do elo de exportação
            exportElo = CSVExportElo()
            
            # Buscar atletas
            athletes = exportElo.getAthletes(athletesIds)
            
            if len(athletes) == 0:
                return False, 'Não há atletas cadastrados para exportar'
            
            # Converter para DataFrame
            df = exportElo.convertAthletesData(athletes)
            
            if not fullData:
                # Exportar apenas CSV
                csv_data = exportElo.generateCSV(df)
                return True, csv_data
            
            # Exportar pacote completo (CSV + gráficos + README)
            csv_data = exportElo.generateCSV(df)
            graphs = exportElo.generateGraphs(df)
            pdf_data = exportElo.generatePDFReport(True)
            zip_data = exportElo.generateZip(csv_data, graphs, pdf_data, includeReadme=True)
            
            return True, zip_data
            
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
            
            # clusterResult é um dicionário, extrair apenas o nome do cluster
            if isinstance(clusterResult, dict):
                athlete.cluster = clusterResult['cluster']
            else:
                athlete.cluster = clusterResult
            
            self.session.add(athlete)
            self.session.commit()
            
            return True, f'Atleta {athlete.nome} cadastrado com sucesso no cluster {athlete.cluster}!'
            
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


    def generatePDFReport(self, includeGraphs: bool = True) -> Tuple[Literal[True, False, -1], bytes | str]:
        """
        Gera relatório PDF completo com todas as análises.
        
        Args:
            includeGraphs: Se deve incluir gráficos no PDF
            
        Returns:
            Tupla (status, pdf_bytes_ou_mensagem)
        """
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from io import BytesIO
            
            # Obter análise completa
            status, data = self.getCompleteAnalysis()
            
            if status != True:
                return status, data
            
            # Criar buffer para PDF
            buffer = BytesIO()
            
            # Criar documento
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                topMargin=2*cm,
                bottomMargin=2*cm,
                leftMargin=2*cm,
                rightMargin=2*cm
            )
            
            # Estilos
            styles = getSampleStyleSheet()
            
            titleStyle = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1e293b'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            headingStyle = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#334155'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            bodyStyle = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                spaceAfter=8
            )
            
            # Elementos do PDF
            elements = []
            
            # Título
            elements.append(Paragraph("RELATÓRIO DE ANÁLISE DE ATLETAS", titleStyle))
            elements.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", bodyStyle))
            elements.append(Spacer(1, 1*cm))
            
            # Resumo Executivo
            elements.append(Paragraph("RESUMO EXECUTIVO", headingStyle))
            
            resumo = data['resumo_executivo']
            
            resumoData = [
                ['Métrica', 'Valor'],
                ['Total de Atletas', str(resumo['total_atletas'])],
                ['Clusters Identificados', str(resumo['clusters'])],
                ['Qualidade Clustering', f"{resumo['qualidade_clustering']:.2f}"],
                ['Anomalias Detectadas', str(resumo['anomalias'])]
            ]
            
            resumoTable = Table(resumoData, colWidths=[8*cm, 8*cm])
            resumoTable.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(resumoTable)
            elements.append(Spacer(1, 0.5*cm))
            
            # Distribuição por Cluster
            elements.append(Paragraph("DISTRIBUIÇÃO POR CLUSTER", headingStyle))
            
            distData = [['Cluster', 'Quantidade', 'Percentual']]
            
            for cluster, qtd in data['visualizacao']['estatisticas']['distribuicao'].items():
                perc = (qtd / resumo['total_atletas']) * 100
                distData.append([cluster, str(qtd), f"{perc:.1f}%"])
            
            distTable = Table(distData, colWidths=[6*cm, 5*cm, 5*cm])
            distTable.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(distTable)
            elements.append(PageBreak())
            
            # Insights Principais
            elements.append(Paragraph("INSIGHTS PRINCIPAIS", headingStyle))
            
            for i, insight in enumerate(data['analise_estatistica']['insights'][:5], 1):
                elements.append(Paragraph(f"<b>{i}. {insight['titulo']}</b>", bodyStyle))
                elements.append(Paragraph(insight['descricao'], bodyStyle))
                elements.append(Paragraph(f"<i>Ação recomendada: {insight['acao']}</i>", bodyStyle))
                elements.append(Spacer(1, 0.3*cm))
            
            elements.append(PageBreak())
            
            # Top Performers
            elements.append(Paragraph("TOP PERFORMERS", headingStyle))
            
            topData = [['Posição', 'Nome', 'Score', 'Cluster']]
            
            for i, performer in enumerate(resumo['top_performers'], 1):
                topData.append([
                    str(i),
                    performer['nome'],
                    f"{performer['score']:.2f}",
                    performer['cluster']
                ])
            
            topTable = Table(topData, colWidths=[2*cm, 6*cm, 4*cm, 4*cm])
            topTable.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(topTable)
            
            # Construir PDF
            doc.build(elements)
            
            # Retornar bytes
            buffer.seek(0)
            return True, buffer.read()
            
        except Exception as e:
            error_msg = (f'{type(e).__name__}: {e} '
                        f'in line {sys.exc_info()[-1].tb_lineno} '
                        f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, error_msg
        
        
    def getAthleteById(self, athleteId: str) -> Tuple[Literal[True, False, -1], Athlete | str]:
        """
        Busca um atleta por ID.
        
        Args:
            athleteId: ID do atleta
            
        Returns:
            Tupla (status, athlete_ou_mensagem)
        """
        try:
            with Config.app.app_context():
                athlete = self.session.query(Athlete).filter_by(id=athleteId).first()
                
                if not athlete:
                    return False, f'Atleta com ID {athleteId} não encontrado'
                
                return True, athlete
                
        except Exception as e:
            error_msg = f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
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