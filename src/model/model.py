import locale, sys, uuid, pandas as pd, pandas as pd, sys, zipfile, datetime, matplotlib.pyplot as plt, io, numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, PageBreak
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter, A4
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from reportlab.lib import colors
from sqlalchemy import func
from typing import Literal

from athleteModel import Athlete
from config import Config

        
class Model:
    def __init__(self):
        self.setupKMeans(n_clusters=3)
        self.CLUSTERS = ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']
        self.COLUMNS = ['nome', 'data_nascimento','sexo', 'altura', 'envergadura', 'arremesso', 'salto_horizontal', 'abdominais']
        self.db = Config.db
        self.session = Config.session
        
        self.create_tables()
        
        
    def setupKMeans(self,  n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters)
        
        
    def loadCSVData(self, file: bytes):
        try:
            data = pd.read_csv(file)
            
            if not self.vrfyCSVColumns(data):
                raise ValueError("Erro ao carregar dados. Verifique se os nomes dos atributos estão corretos.")
            
            data = self.sanitizeCSV(data) 
            
            data = data.to_dict(orient='records')
            
            return self.createAthletes(data)
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        

    def vrfyCSVColumns(self, data: pd.DataFrame) -> bool:
        try:
            frameColumns = data.columns
            for column in self.COLUMNS:
                col = frameColumns[column.lower()]
        except (KeyError, AttributeError):
            return False


    def sanitizeCSV(self, data: pd.DataFrame) -> pd.DataFrame:
        def isNum(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return False
        
        def isDate(x):
            try:
                return datetime.strptime(str(x), '%Y-%m-%d')
            except (ValueError, TypeError):
                return False
        
        def isSex(x):
            if str(x).upper() == 'M':
                return 0
            if str(x).upper() == 'F':
                return 1
            return False
        
        data = data.copy()
        
        numColumns = [col for col in self.COLUMNS 
                    if col not in ['nome', 'data_nascimento', 'sexo']]
        
        for column in numColumns:
            converted = data[column].apply(isNum)
            if (converted == False).any():
                raise ValueError(f"Coluna '{column}' contém valores não numéricos")
            data[column] = converted
        
        converted_dates = data['data_nascimento'].apply(isDate)
        if (converted_dates == False).any():
            raise ValueError("Coluna 'data_nascimento' contém datas inválidas")
        data['data_nascimento'] = converted_dates
        
        converted_sex = data['sexo'].apply(isSex)
        if (converted_sex == False).any():
            raise ValueError("Coluna 'sexo' contém valores inválidos")
        data['sexo'] = converted_sex
        
        return data
        
    def trainModel(self, nClusters=None):
        """Treina o modelo KMeans com os dados dos atletas"""
        try:
            athletes = self.session.query(Athlete).all()
            
            if len(athletes) == 0:
                return False, 'Não há atletas cadastrados para treinar o modelo'
            
            # Preparar dados
            data = []
            for athlete in athletes:
                data.append([
                    athlete.estatura,
                    athlete.envergadura,
                    athlete.arremesso,
                    athlete.saltoHorizontal,
                    athlete.abdominais
                ])
            
            self.data = np.array(data)
            
            # Definir número de clusters
            if nClusters is None:
                nClusters = min(4, len(athletes))
            
            # Padronizar dados
            self.dataScaled = self.scaler.fit_transform(self.data)
            
            # Treinar KMeans
            self.kmeans = KMeans(n_clusters=nClusters, random_state=42)
            self.labels = self.kmeans.fit_predict(self.dataScaled)
            
            # Atualizar clusters dos atletas
            for i, athlete in enumerate(athletes):
                clusterIdx = self.labels[i]
                athlete.cluster = self.CLUSTERS[clusterIdx] if clusterIdx < len(self.CLUSTERS) else f'Cluster {clusterIdx}'
            
            self.session.commit()
            
            return True, 'Modelo treinado com sucesso'
            
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}')
    
    
    def predictCluster(self, athleteData: dict):
        """Prediz o cluster de um novo atleta"""
        try:
            if self.kmeans is None:
                return False, 'Modelo não treinado'
            
            # Preparar dados
            features = np.array([[
                athleteData['estatura'],
                athleteData['envergadura'],
                athleteData['arremesso'],
                athleteData['saltoHorizontal'],
                athleteData['abdominais']
            ]])
            
            # Padronizar e predizer
            featuresScaled = self.scaler.transform(features)
            clusterIdx = self.kmeans.predict(featuresScaled)[0]
            probabilities = self.getClusterProbabilities(featuresScaled)
            
            clusterName = self.CLUSTERS[clusterIdx] if clusterIdx < len(self.CLUSTERS) else f'Cluster {clusterIdx}'
            
            return True, {
                'cluster': clusterName,
                'clusterIdx': int(clusterIdx),
                'probabilities': probabilities
            }
            
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}')
    
    
    def getClusterProbabilities(self, featuresScaled):
        """Calcula probabilidades de pertencer a cada cluster baseado na distância"""
        distances = self.kmeans.transform(featuresScaled)[0]
        invDistances = 1 / (distances + 1e-10)
        probabilities = invDistances / invDistances.sum()
        
        result = {}
        for i, prob in enumerate(probabilities):
            clusterName = self.CLUSTERS[i] if i < len(self.CLUSTERS) else f'Cluster {i}'
            result[clusterName] = float(prob * 100)
        
        return result
    
    
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
        
        
    def exportData(self, fullData: bool = False) -> tuple[bool, bytes] | tuple[bool, dict]:
        try:
            data = self.getAthletes(paginated=False)
                
            if len(data) == 0:
                return False, 'Não há dados para exportar'
            
            dataCSV = self.generateCSV(data)
            
            if fullData:
                files = [dataCSV]
                filenames = ["athletes.csv"]
                
                stats, graphcs = self.getViewInfo()
                if stats != True:
                    return stats, graphcs
                
                for graph in graphcs:
                    graph = pd.DataFrame(graph)
                    files.append(graph)    
                
                compression = zipfile.ZIP_DEFLATED
                
                zf = zipfile.ZipFile('fullData.zip', 'w')
                try:
                    i=0
                    for file in files:
                        zf.write(file, filenames[i], compression=compression)
                        i+=1
                except Exception as e:
                    return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
                finally:
                    zf.close()
            else:
                data = self.model.getAthletes(paginated=False)
                
                if len(data) == 0:
                    return False, 'Não há dados para exportar'
                
                dataCSV = self.generateCSV(data)
                return True, dataCSV
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
        
    def create_tables(self) -> None:
        with Config.app.app_context():
            self.db.create_all()
    
    
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
    
    
    def createAthletes(self, data: list[pd.DataFrame] | dict) -> bool:
        if type(data) == list:
            if len(data) == 0:
                raise ValueError("Não há dados para exportar")
            
            for item in data:
                athlete = Athlete(nome=item['nome'], data_nascimento=item['data_nascimento'], sexo=item['sexo'], 
                                estatura=item['estatura'], envergadura=item['envergadura'], 
                                arremesso=item['arremesso'], salto_horizontal=item['salto_horizontal'], 
                                abdominais=item['abdominais'])
                self.session.add(athlete)
            
            self.session.commit()
            return True
        if type(data) == dict:
            athlete = Athlete(nome=data['nome'], data_nascimento=data['data_nascimento'], sexo=data['sexo'], 
                            estatura=data['estatura'], envergadura=data['envergadura'], 
                            arremesso=data['arremesso'], salto_horizontal=data['salto_horizontal'], 
                            abdominais=data['abdominais'])
            self.session.add(athlete)
            
            self.session.commit()
            return True
        
        raise ValueError("Erro ao criar athlete")
    
    
    def exportData(self, fullData: bool = False) -> tuple[bool, bytes] | tuple[bool, bytes]:
        data = self.getAthletes(paginated=False)
            
        if len(data) == 0:
            return False, 'Não há dados para exportar'
        
        if fullData:
            modelHealth = self.getModelHealth()
            dbStats = self.getDBStats()
            
            graphs = self.genGraphs(modelHealth, dbStats)
            
            pdf = self.genPDF(data, graphs)
            
            return True, pdf
            
        csv = self.generateCSV(data)
        return True, csv
        
        
    def getModelHealth(self) -> dict: # TODO: Review this function
        # Assumindo que você tem:
        # self.kmeans = seu modelo KMeans/clustering
        # self.data = seus dados (X)
        # self.labels = labels dos clusters (ou self.kmeans.labels_)
        
        labels = self.kmeans.labels_  # ou self.labels
        X = self.data  # seus dados
        
        # 1. Silhouette Score (geral e por amostra)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_samples_values = silhouette_samples(X, labels)
        
        # Organizar silhouette por cluster para gráfico
        silhouette_by_cluster = {}
        for i in range(self.kmeans.n_clusters):
            cluster_silhouette_values = silhouette_samples_values[labels == i]
            silhouette_by_cluster[f'cluster_{i}'] = {
                'values': cluster_silhouette_values.tolist(),
                'mean': float(np.mean(cluster_silhouette_values)),
                'size': int(np.sum(labels == i))
            }
        
        # 2. Davies-Bouldin Index (menor é melhor)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        # 3. Inércia Total (Within-Cluster Sum of Squares)
        inertia = self.kmeans.inertia_
        
        # Inércia por cluster (opcional, mas útil para gráficos)
        inertia_by_cluster = {}
        for i in range(self.kmeans.n_clusters):
            cluster_points = X[labels == i]
            centroid = self.kmeans.cluster_centers_[i]
            cluster_inertia = np.sum((cluster_points - centroid) ** 2)
            inertia_by_cluster[f'cluster_{i}'] = float(cluster_inertia)
        
        # 4. PCA para visualização dos clusters
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Organizar dados PCA por cluster
        pca_data = {
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(np.sum(pca.explained_variance_ratio_)),
            'clusters': {}
        }
        
        for i in range(self.kmeans.n_clusters):
            cluster_mask = labels == i
            pca_data['clusters'][f'cluster_{i}'] = {
                'x': X_pca[cluster_mask, 0].tolist(),
                'y': X_pca[cluster_mask, 1].tolist(),
                'size': int(np.sum(cluster_mask))
            }
        
        # Adicionar centroides no espaço PCA
        centroids_pca = pca.transform(self.kmeans.cluster_centers_)
        pca_data['centroids'] = {
            'x': centroids_pca[:, 0].tolist(),
            'y': centroids_pca[:, 1].tolist()
        }
        
        health_metrics = {
            'silhouette': {
                'score': float(silhouette_avg),
                'by_cluster': silhouette_by_cluster,
                'interpretation': 'good' if silhouette_avg > 0.5 else 'moderate' if silhouette_avg > 0.25 else 'poor'
            },
            'davies_bouldin': {
                'score': float(davies_bouldin),
                'interpretation': 'good' if davies_bouldin < 1.0 else 'moderate' if davies_bouldin < 2.0 else 'poor'
            },
            'inertia': {
                'total': float(inertia),
                'by_cluster': inertia_by_cluster
            },
            'pca': pca_data,
            'n_clusters': int(self.kmeans.n_clusters),
            'n_samples': int(len(X))
        }
        
        return health_metrics
    
    
    def getViewInfo(self) -> tuple[bool | Literal[-1], dict | str]:
        try:
            dbStats = self.getDBStats()
            modelHealth = self.getModelHealth()
            
            return True, {
                'dbStats': dbStats,
                'modelHealth': modelHealth
            }
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
        
    def getDBStats(self) -> dict:
        ...
        
    
    def genPDF(self, data: list[dict], graphs: list[bytes]) -> bytes:
        ...
        
        
    def genGraphs(self) -> list[bytes]:
        ...
    
        
    def generateCSV(self, data: list[dict]) -> bytes:
        dataCSV = pd.DataFrame(data).to_csv(index=False)
        return dataCSV.encode('utf-8')