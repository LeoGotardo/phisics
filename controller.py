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
import pickle as pkl
import pandas as pd
import numpy as np
import sys


class Controller:
    def __init__(self):
        self.model = Model()
        self.setupKMeans(n_clusters=4)
        
        
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
            columns = ['nome', 'data_nascimento', 'sexo', 'massa_corporal', 'estatura', 'envergadura', 'arremesso', 'salto_horizontal', 'abdominais']
            
            if not all(column in athletes.columns for column in columns) or len(athletes) == 0 or athletes.isnull().values.any() or len(athletes.columns) != len(columns):
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
            return status, info
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')

    
    def getAthlets(self, dados):
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

    
    def getCluster(self, clusterId):
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

    
    def createCluster(self, dados):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')

    
    def getClusterInfo(self, clusterId):
        try:
            pass
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno} in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
