import pandas as pd
import numpy as np
from icecream import ic

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from typing import List, Dict

from src.model.athleteModel import Athlete
from src.utils.dataclasses import Elo
from src.config import Config


class ViewDataElo(Elo):
    """
    Pipeline para gerar dados estatísticos para a página de visualização.
    Processa dados de atletas através de uma cadeia de transformações.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self.features = ['sexo_encoded', 'altura', 'envergadura', 
                        'arremesso', 'saltoHorizontal', 'abdominais']
        
        self.clusterNames = {
            0: 'Iniciante', 
            1: 'Intermediário', 
            2: 'Competitivo', 
            3: 'Elite'
        }
        
        self._buildChain()
    
    
    def _buildChain(self) -> None:
        """Constrói a cadeia de processamento dos dados"""
        
        self.chain = [
            self.getAthletes,
            self.prepareDataFrame,
            self.generatePcaData,
            self.generateCorrelacaoData,
            self.generateCoreData,
            self.generateRadarData,
            self.generateMetricas,
            self.generateEstatisticas,
            self.assembleResult
        ]
    
    
    def getAthletes(self, _: None = None) -> List[Athlete]:
        """
        Busca todos os atletas do banco de dados.
        
        Args:
            _: Parâmetro ignorado (para iniciar a cadeia)
            
        Returns:
            Lista de objetos Athlete
        """
        with Config.app.app_context():
            athletes = Config.session.query(Athlete).filter(Athlete.cluster != -1).all()
        
        if len(athletes) == 0:
            raise ValueError("Nenhum atleta cadastrado. Cadastre atletas para visualizar análises.")
        
        if len(athletes) < 10:
            raise ValueError(f"Necessário pelo menos 10 atletas para análise estatística. Cadastrados: {len(athletes)}")
        
        return athletes
    
    
    def prepareDataFrame(self, athletes: List[Athlete]) -> Dict:
        """
        Converte lista de atletas em DataFrame preparado.
        
        Args:
            athletes: Lista de objetos Athlete
            
        Returns:
            Dicionário com DataFrame e dados originais
        """
        athletesData = [athlete.dict() for athlete in athletes]
        df = pd.DataFrame(athletesData)
        
        if 'sexo' in df.columns:
            df['sexo_encoded'] = df['sexo'].map({'M': 1, 'F': 0})
        
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Colunas faltando: {', '.join(missing)}")
        
        df = df.dropna(subset=self.features)
        
        if df['cluster'].dtype == 'object':
            reverseMapping = {v: k for k, v in self.clusterNames.items()}
            df['clusterId'] = df['cluster'].map(reverseMapping)
        else:
            df['clusterId'] = df['cluster']
        
        return {
            'df': df,
            'athletes': athletes,
            'results': {}
        }
    
    
    def generatePcaData(self, data: Dict) -> Dict:
        """
        Gera dados para o gráfico PCA (scatter plot).
        
        Args:
            data: Dicionário com DataFrame
            
        Returns:
            Dicionário atualizado com dados PCA
        """
        df = data['df']
        
        X = df[self.features].values
        scaler = StandardScaler()
        xScaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        xPca = pca.fit_transform(xScaled)
        
        df['PC1'] = xPca[:, 0]
        df['PC2'] = xPca[:, 1]
        
        pcaData = []
        
        for clusterId in sorted(df['clusterId'].unique()):
            clusterName = self.clusterNames.get(int(clusterId), f'Cluster {clusterId}')
            dfCluster = df[df['clusterId'] == clusterId]
            
            pcaData.append({
                'nivel': clusterName,
                'x': dfCluster['PC1'].round(2).tolist(),
                'y': dfCluster['PC2'].round(2).tolist()
            })
        
        varianceExplained = {
            'pc1': round(pca.explained_variance_ratio_[0] * 100, 1),
            'pc2': round(pca.explained_variance_ratio_[1] * 100, 1),
            'total': round(sum(pca.explained_variance_ratio_) * 100, 1)
        }
        
        data['results']['pca_data'] = pcaData
        data['results']['variance_explained'] = varianceExplained
        data['df'] = df
        
        return data
    
    
    def generateCorrelacaoData(self, data: Dict) -> Dict:
        """
        Gera dados para o gráfico de correlação.
        
        Args:
            data: Dicionário com DataFrame
            
        Returns:
            Dicionário atualizado com dados de correlação
        """
        df = data['df']
        
        potenciaSuperior = df['arremesso'].values
        potenciaInferior = df['saltoHorizontal'].values
        
        rValue, pValue = pearsonr(potenciaSuperior, potenciaInferior)
        
        correlacaoData = {
            'x': [round(x, 2) for x in potenciaSuperior.tolist()],
            'y': [round(y, 2) for y in potenciaInferior.tolist()],
            'correlacao': round(rValue, 2),
            'p_value': '0.001' if pValue < 0.001 else f'{pValue:.3f}'
        }
        
        data['results']['potencia_data'] = correlacaoData
        
        return data
    
    
    def generateCoreData(self, data: Dict) -> Dict:
        """
        Gera dados para o gráfico de força do core.
        
        Args:
            data: Dicionário com DataFrame
            
        Returns:
            Dicionário atualizado com dados de core
        """
        df = data['df']
        coreData = []
        
        for clusterId in sorted(df['clusterId'].unique()):
            clusterName = self.clusterNames.get(int(clusterId), f'Cluster {clusterId}')
            dfCluster = df[df['clusterId'] == clusterId]
            
            # CORREÇÃO: Remover NaN antes de calcular estatísticas
            abdominais = dfCluster['abdominais'].dropna()
            
            if len(abdominais) == 0:
                continue
            
            coreData.append({
                'nivel': clusterName,
                'media': round(abdominais.mean(), 1),
                'std': round(abdominais.std(), 1),
                'min': int(abdominais.min()),
                'max': int(abdominais.max())
            })
        
        data['results']['core_data'] = coreData
        
        return data
    
    
    def generateRadarData(self, data: Dict) -> Dict:
        """
        Gera dados para o gráfico radar.
        
        Args:
            data: Dicionário com DataFrame
            
        Returns:
            Dicionário atualizado com dados radar
        """
        df = data['df']
        radarFeatures = ['arremesso', 'saltoHorizontal', 'abdominais', 'altura', 'envergadura']
        
        dfNormalized = df.copy()
        for feat in radarFeatures:
            minVal = df[feat].min()
            maxVal = df[feat].max()
            if maxVal > minVal:
                dfNormalized[f'{feat}_norm'] = ((df[feat] - minVal) / (maxVal - minVal)) * 100
            else:
                dfNormalized[f'{feat}_norm'] = 50
        
        radarData = []
        
        for clusterId in sorted(dfNormalized['clusterId'].unique()):
            clusterName = self.clusterNames.get(int(clusterId), f'Cluster {clusterId}')
            dfCluster = dfNormalized[dfNormalized['clusterId'] == clusterId]
            
            valores = [
                round(dfCluster[f'{feat}_norm'].mean(), 1) 
                for feat in radarFeatures
            ]
            
            radarData.append({
                'nivel': clusterName,
                'features': radarFeatures,
                'valores': valores
            })
        
        data['results']['perfil_data'] = radarData
        
        return data
    
    
    def generateMetricas(self, data: Dict) -> Dict:
        """
        Calcula métricas de qualidade do clustering.
        
        Args:
            data: Dicionário com DataFrame
            
        Returns:
            Dicionário atualizado com métricas
        """
        df = data['df']
        
        X = df[self.features].values
        scaler = StandardScaler()
        xScaled = scaler.fit_transform(X)
        
        labels = df['clusterId'].values
        
        silhouette = silhouette_score(xScaled, labels)
        daviesBouldin = davies_bouldin_score(xScaled, labels)
        
        nClusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=nClusters, random_state=42, n_init=10)
        kmeans.fit(xScaled)
        inertia = kmeans.inertia_
        
        metricas = {
            'silhouette': round(silhouette, 2),
            'davies_bouldin': round(daviesBouldin, 2),
            'inertia': round(inertia, 0)
        }
        
        data['results']['metricas'] = metricas
        
        return data
    
    
    def generateEstatisticas(self, data: Dict) -> Dict:
        """
        Gera estatísticas gerais dos dados.
        
        Args:
            data: Dicionário com DataFrame
            
        Returns:
            Dicionário atualizado com estatísticas
        """
        df = data['df']
        
        distribuicao = {}
        for clusterId in sorted(df['clusterId'].unique()):
            clusterName = self.clusterNames.get(int(clusterId), f'Cluster {clusterId}')
            count = len(df[df['clusterId'] == clusterId])
            distribuicao[clusterName] = count
        
        estatisticas = {
            'total_atletas': len(df),
            'distribuicao': distribuicao,
            'features_analisadas': len(self.features)
        }
        
        data['results']['estatisticas'] = estatisticas
        
        return data
    
    
    def assembleResult(self, data: Dict) -> Dict:
        """
        Monta o resultado final com todos os dados.
        
        Args:
            data: Dicionário com todos os resultados
            
        Returns:
            Dicionário final com estrutura limpa
        """
        return data['results']