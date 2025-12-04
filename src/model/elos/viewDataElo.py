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
        
        self.cluster_names = {
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
            self.generatePCAData,
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
        athletes_data = [athlete.dict() for athlete in athletes]
        df = pd.DataFrame(athletes_data)
        
        if 'sexo' in df.columns:
            df['sexo_encoded'] = df['sexo'].map({'M': 1, 'F': 0})
        
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Colunas faltando: {', '.join(missing)}")
        
        df = df.dropna(subset=self.features)
        
        if df['cluster'].dtype == 'object':
            reverse_mapping = {v: k for k, v in self.cluster_names.items()}
            df['cluster_id'] = df['cluster'].map(reverse_mapping)
        else:
            df['cluster_id'] = df['cluster']
        
        return {
            'df': df,
            'athletes': athletes,
            'results': {}
        }
    
    
    def generatePCAData(self, data: Dict) -> Dict:
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
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        df['PC1'] = X_pca[:, 0]
        df['PC2'] = X_pca[:, 1]
        
        pca_data = []
        
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_name = self.cluster_names.get(int(cluster_id), f'Cluster {cluster_id}')
            df_cluster = df[df['cluster_id'] == cluster_id]
            
            pca_data.append({
                'nivel': cluster_name,
                'x': df_cluster['PC1'].round(2).tolist(),
                'y': df_cluster['PC2'].round(2).tolist()
            })
        
        variance_explained = {
            'pc1': round(pca.explained_variance_ratio_[0] * 100, 1),
            'pc2': round(pca.explained_variance_ratio_[1] * 100, 1),
            'total': round(sum(pca.explained_variance_ratio_) * 100, 1)
        }
        
        data['results']['pca_data'] = pca_data
        data['results']['variance_explained'] = variance_explained
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
        
        potencia_superior = df['arremesso'].values
        potencia_inferior = df['saltoHorizontal'].values
        
        r_value, p_value = pearsonr(potencia_superior, potencia_inferior)
        
        correlacao_data = {
            'x': [round(x, 2) for x in potencia_superior.tolist()],
            'y': [round(y, 2) for y in potencia_inferior.tolist()],
            'correlacao': round(r_value, 2),
            'p_value': '0.001' if p_value < 0.001 else f'{p_value:.3f}'
        }
        
        data['results']['potencia_data'] = correlacao_data
        
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
        core_data = []
        
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_name = self.cluster_names.get(int(cluster_id), f'Cluster {cluster_id}')
            df_cluster = df[df['cluster_id'] == cluster_id]
            
            abdominais = df_cluster['abdominais']
            
            ic(abdominais)
            
            core_data.append({
                'nivel': cluster_name,
                'media': round(abdominais.mean(), 1),
                'std': round(abdominais.std(), 1),
                'min': int(abdominais.min()),
                'max': int(abdominais.max())
            })
        
        data['results']['core_data'] = core_data
        
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
        radar_features = ['arremesso', 'saltoHorizontal', 'abdominais', 'altura', 'envergadura']
        
        df_normalized = df.copy()
        for feat in radar_features:
            min_val = df[feat].min()
            max_val = df[feat].max()
            if max_val > min_val:
                df_normalized[f'{feat}_norm'] = ((df[feat] - min_val) / (max_val - min_val)) * 100
            else:
                df_normalized[f'{feat}_norm'] = 50
        
        radar_data = []
        
        for cluster_id in sorted(df_normalized['cluster_id'].unique()):
            cluster_name = self.cluster_names.get(int(cluster_id), f'Cluster {cluster_id}')
            df_cluster = df_normalized[df_normalized['cluster_id'] == cluster_id]
            
            valores = [
                round(df_cluster[f'{feat}_norm'].mean(), 1) 
                for feat in radar_features
            ]
            
            radar_data.append({
                'nivel': cluster_name,
                'features': radar_features,
                'valores': valores
            })
        
        data['results']['perfil_data'] = radar_data
        
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
        X_scaled = scaler.fit_transform(X)
        
        labels = df['cluster_id'].values
        
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia = kmeans.inertia_
        
        metricas = {
            'silhouette': round(silhouette, 2),
            'davies_bouldin': round(davies_bouldin, 2),
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
        for cluster_id in sorted(df['cluster_id'].unique()):
            cluster_name = self.cluster_names.get(int(cluster_id), f'Cluster {cluster_id}')
            count = len(df[df['cluster_id'] == cluster_id])
            distribuicao[cluster_name] = count
        
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