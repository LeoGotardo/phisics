"""
TODO:
    - Setup Kmeans
    - Criar cluster
    - Criar modelo
    - Carregar dados
    - Analisar dados
    - Criar cadeia de responsavilidades
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from model import Model

import pickle as pkl, pandas as pd, sys, zipfile


class Controller:
    def __init__(self):
        self.model = Model()
        self.setupKMeans(n_clusters=3)
        self.CLUSTERS = ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']
        self.COLUMNS = ['nome', 'data_nascimento','sexo', 'altura', 'envergadura', 'arremesso', 'salto_horizontal', 'abdominais']
        
        
    def setupKMeans(self,  n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters)
        
        
    def loadcsvData(self, file: bytes):
        try:
            def isNum(x):
                try:
                    float(x)
                    return True
                except ValueError:
                    return False
            
            
            athletes = pd.read_csv(file)
            if not all(column in athletes.columns for column in self.COLUMNS) or len(athletes) == 0 or athletes.isnull().values.any() or len(athletes.columns) != len(self.COLUMNS):
                return False, 'Erro ao carregar dados. Verifique se os nomes dos atributos estão corretos.'
            
            for column in athletes.columns:
                if column not in ['nome', 'sexo', 'data_nascimento']:
                    is_numeric = athletes[column].apply(isNum)
                    if not is_numeric.all():
                        return False, f'Erro ao carregar dados. O campo "{column}" contém valores não numéricos.'
            
            
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
        
    def getViewInfo(self):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
    
    def getDashboardInfo(self):
        try:
            status, info = self.model.getDashboardInfo()
            if status == True:
                labels = self.kmeans.labels_
                count = pd.Series(labels).value_counts().sort_index()
                
                centroides = self.kmeans.cluster_centers_
                validColumns = self.COLUMNS
                validColumns.remove('nome').remove('data_nascimento')
                
                for i, centroide in enumerate(centroides):
                    for column in validColumns:
                        info['clusterInfo'][i][column] = centroide[i]
                    info['clusterInfo'][i] = {
                        'label': self.CLUSTERS[i],
                        'count': count[i]
                    }
                
                return status, info
            else:
                return status, str(info)
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
        
    def exportData(self, fullData: bool = False) -> tuple[bool, bytes] | tuple[bool, dict]:
        try:
            data = self.model.getAthletes(paginated=False)
                
            if len(data) == 0:
                return False, 'Não há dados para exportar'
            
            dataCSV = pd.DataFrame(data).to_csv(index=False)
            
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
                
                dataCSV = pd.DataFrame(data).to_csv(index=False)
                return True, dataCSV
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')

    
    def getAthlete(self, dados):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')

    
    def putAthlete(self, dados):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')

    
    def listAthletes(self, filtros):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')

    
    def getModelQuality(self):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')

    
    def getModelMetrics(self):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
        
    
    def getClusterInfo(self, clusterId):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
