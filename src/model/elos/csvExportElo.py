import pandas as pd

from utils.dataclasses import Column, Elo
from model.athleteModel import Athlete
from utils.dataUtils import DataUtils
from typing import Callable, Any
from config import Config
from model import Model

class CSVExportElo(Elo):
    """
    Pipeline para exportação de dados CSV com cada coluna validada.
    Herda de Elo e implementa a cadeia de processamento.
    """
    
    def __init__(self) -> None:
        # Inicializar a cadeia vazia (do Elo pai)
        
        super().__init__()
        
        self.COLUMNS: list[Column] = Config.COLUMNS
        
        self._build_chain()
        
    
    def _build_chain(self) -> None:
        """Constrói a cadeia de funções para gerar o CSV"""
        
        self.chain = [
            self.getAthletes,
            self.convertAthletesData,
            self.generateCSV
        ]
        
    
    def getAthletes(self, paginated: bool = False) -> list[Athlete]:
        """
        Lê os atletas da base de dados.
        
        Args:
            paginated: Se True, retorna uma lista paginada
            
        Returns:
            Lista de atletas
        """
   
   
#TODO: Implementar isso     
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
                data = self.getAthletes(paginated=False)
                
                if len(data) == 0:
                    return False, 'Não há dados para exportar'
                
                dataCSV = self.generateCSV(data)
                return True, dataCSV
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        

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
        