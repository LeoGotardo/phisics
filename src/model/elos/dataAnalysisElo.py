import pandas as pd, numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from src.model.athleteModel import Athlete
from sklearn.decomposition import PCA
from src.utils.dataclasses import Elo
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from src.config import Config
from scipy import stats


class DataAnalysisElo(Elo):
    """
    Pipeline para análise estatística avançada dos dados de atletas.
    Gera insights, correlações e análises descritivas.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self.clusterNames = {
            0: 'Iniciante',
            1: 'Intermediário',
            2: 'Competitivo',
            3: 'Elite'
        }
        
        self.features = ['altura', 'envergadura', 'arremesso', 
                        'saltoHorizontal', 'abdominais']
        
        self._buildChain()
    
    
    def _buildChain(self) -> None:
        """Constrói a cadeia de análises"""
        
        self.chain = [
            self.getAthletes,
            self.prepareDataFrame,
            self.performDescriptiveAnalysis,
            self.performCorrelationAnalysis,
            self.performClusterAnalysis,
            self.performANOVAAnalysis,
            self.performRegressionAnalysis,
            self.generateInsights,
            self.assembleAnalysisReport
        ]
    
    
    def getAthletes(self, _: None = None) -> List[Athlete]:
        """Busca todos os atletas do banco"""
        
        with Config.app.app_context():
            athletes = Config.session.query(Athlete).all()
        
        if len(athletes) == 0:
            raise ValueError("Nenhum atleta cadastrado.")
        
        if len(athletes) < 10:
            raise ValueError(f"Necessário pelo menos 10 atletas. Cadastrados: {len(athletes)}")
        
        return athletes
    
    
    def prepareDataFrame(self, athletes: List[Athlete]) -> Dict:
        """Converte atletas em DataFrame preparado"""
        
        data = [athlete.dict() for athlete in athletes]
        df = pd.DataFrame(data)
        
        # Codificar sexo
        df['sexo_encoded'] = df['sexo'].map({'M': 1, 'F': 0})
        
        # Calcular idade
        df['idade'] = pd.to_datetime('today').year - pd.to_datetime(df['dataNascimento']).dt.year
        
        # Mapear clusters
        if df['cluster'].dtype == 'object':
            reverseMapping = {v: k for k, v in self.clusterNames.items()}
            df['cluster_id'] = df['cluster'].map(reverseMapping)
        else:
            df['cluster_id'] = df['cluster']
        
        # Calcular score composto
        df['score_total'] = (
            df['arremesso'] * 0.3 + 
            df['saltoHorizontal'] * 0.3 + 
            df['abdominais'] * 0.2 +
            df['altura'] * 0.1 +
            df['envergadura'] * 0.1
        )
        
        return {
            'df': df,
            'athletes': athletes,
            'analysis': {}
        }
    
    
    def performDescriptiveAnalysis(self, data: Dict) -> Dict:
        """Análise estatística descritiva por cluster"""
        
        df = data['df']
        
        descriptive = {}
        
        for clusterId in sorted(df['cluster_id'].unique()):
            clusterName = self.clusterNames[int(clusterId)]
            dfCluster = df[df['cluster_id'] == clusterId]
            
            stats_dict = {}
            
            for feature in self.features:
                stats_dict[feature] = {
                    'media': float(dfCluster[feature].mean()),
                    'mediana': float(dfCluster[feature].median()),
                    'desvio': float(dfCluster[feature].std()),
                    'min': float(dfCluster[feature].min()),
                    'max': float(dfCluster[feature].max()),
                    'q1': float(dfCluster[feature].quantile(0.25)),
                    'q3': float(dfCluster[feature].quantile(0.75))
                }
            
            descriptive[clusterName] = {
                'count': len(dfCluster),
                'stats': stats_dict,
                'idade_media': float(dfCluster['idade'].mean()),
                'sexo_dist': {
                    'masculino': int((dfCluster['sexo'] == 'M').sum()),
                    'feminino': int((dfCluster['sexo'] == 'F').sum())
                }
            }
        
        data['analysis']['descriptive'] = descriptive
        return data
    
    
    def performCorrelationAnalysis(self, data: Dict) -> Dict:
        """Análise de correlação entre variáveis"""
        
        df = data['df']
        
        # Matriz de correlação geral
        corrMatrix = df[self.features + ['sexo_encoded', 'idade']].corr()
        
        correlations = {
            'matriz_geral': corrMatrix.to_dict(),
            'pares_significativos': []
        }
        
        # Identificar correlações significativas
        for i, feat1 in enumerate(self.features):
            for feat2 in self.features[i+1:]:
                corr = corrMatrix.loc[feat1, feat2]
                if abs(corr) > 0.5:  # Correlação moderada/forte
                    correlations['pares_significativos'].append({
                        'var1': feat1,
                        'var2': feat2,
                        'correlacao': float(corr),
                        'interpretacao': self._interpretCorrelation(corr)
                    })
        
        # Correlações por cluster
        correlations['por_cluster'] = {}
        
        for clusterId in sorted(df['cluster_id'].unique()):
            clusterName = self.clusterNames[int(clusterId)]
            dfCluster = df[df['cluster_id'] == clusterId]
            
            if len(dfCluster) > 3:  # Mínimo para correlação válida
                corrCluster = dfCluster[self.features].corr()
                correlations['por_cluster'][clusterName] = corrCluster.to_dict()
        
        data['analysis']['correlations'] = correlations
        return data
    
    
    def performClusterAnalysis(self, data: Dict) -> Dict:
        """Análise de qualidade e separação dos clusters"""
        
        df = data['df']
        
        X = df[self.features + ['sexo_encoded']].values
        scaler = StandardScaler()
        XScaled = scaler.fit_transform(X)
        
        labels = df['cluster_id'].values
        
        # Métricas de qualidade
        silhouette = silhouette_score(XScaled, labels)
        daviesBouldin = davies_bouldin_score(XScaled, labels)
        
        # Reconstruir KMeans para obter inércia
        nClusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=nClusters, random_state=42, n_init=10)
        kmeans.fit(XScaled)
        inertia = kmeans.inertia_
        
        # PCA para visualização
        pca = PCA(n_components=2)
        XPca = pca.fit_transform(XScaled)
        
        df['PC1'] = XPca[:, 0]
        df['PC2'] = XPca[:, 1]
        
        # Distâncias intra-cluster
        intraClusterDist = {}
        
        for clusterId in sorted(df['cluster_id'].unique()):
            clusterName = self.clusterNames[int(clusterId)]
            mask = df['cluster_id'] == clusterId
            clusterPoints = XScaled[mask]
            
            if len(clusterPoints) > 1:
                centroid = clusterPoints.mean(axis=0)
                distances = np.linalg.norm(clusterPoints - centroid, axis=1)
                
                intraClusterDist[clusterName] = {
                    'media': float(distances.mean()),
                    'desvio': float(distances.std()),
                    'max': float(distances.max())
                }
        
        clusterAnalysis = {
            'qualidade': {
                'silhouette': float(silhouette),
                'davies_bouldin': float(daviesBouldin),
                'inertia': float(inertia)
            },
            'variancia_explicada': {
                'pc1': float(pca.explained_variance_ratio_[0] * 100),
                'pc2': float(pca.explained_variance_ratio_[1] * 100),
                'total': float(sum(pca.explained_variance_ratio_) * 100)
            },
            'distancias_intra_cluster': intraClusterDist,
            'interpretacao': self._interpretClusterQuality(silhouette, daviesBouldin)
        }
        
        data['analysis']['cluster_analysis'] = clusterAnalysis
        data['df'] = df
        
        return data
    
    
    def performANOVAAnalysis(self, data: Dict) -> Dict:
        """ANOVA para testar diferenças entre clusters"""
        
        df = data['df']
        
        anovaResults = {}
        
        for feature in self.features:
            groups = [df[df['cluster_id'] == i][feature].dropna() 
                     for i in sorted(df['cluster_id'].unique())]
            
            # ANOVA de um fator
            fStat, pValue = stats.f_oneway(*groups)
            
            anovaResults[feature] = {
                'f_statistic': float(fStat),
                'p_value': float(pValue),
                'significativo': pValue < 0.05,
                'interpretacao': self._interpretANOVA(pValue)
            }
        
        data['analysis']['anova'] = anovaResults
        return data
    
    
    def performRegressionAnalysis(self, data: Dict) -> Dict:
        """Análise de regressão para predição de desempenho"""
        
        df = data['df']
        
        from scipy.stats import linregress
        
        regressions = {}
        
        # Regressão: altura vs performance
        slope, intercept, rValue, pValue, stdErr = linregress(
            df['altura'], df['score_total']
        )
        
        regressions['altura_performance'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(rValue ** 2),
            'p_value': float(pValue),
            'interpretacao': f"A cada 1cm de altura, o score aumenta {slope:.2f} pontos"
        }
        
        # Regressão: idade vs performance
        slope, intercept, rValue, pValue, stdErr = linregress(
            df['idade'], df['score_total']
        )
        
        regressions['idade_performance'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(rValue ** 2),
            'p_value': float(pValue),
            'interpretacao': f"A cada ano de idade, o score muda {slope:.2f} pontos"
        }
        
        data['analysis']['regressions'] = regressions
        return data
    
    
    def generateInsights(self, data: Dict) -> Dict:
        """Gera insights acionáveis baseados nas análises"""
        
        df = data['df']
        analysis = data['analysis']
        
        insights = []
        
        # Insight 1: Cluster dominante
        clusterCounts = df['cluster_id'].value_counts()
        dominantCluster = self.clusterNames[clusterCounts.idxmax()]
        
        insights.append({
            'tipo': 'distribuicao',
            'titulo': 'Cluster Dominante',
            'descricao': f"O cluster '{dominantCluster}' representa {(clusterCounts.max() / len(df) * 100):.1f}% dos atletas",
            'acao': f"Considere programas específicos para desenvolvimento do cluster '{dominantCluster}'"
        })
        
        # Insight 2: Feature mais discriminante
        anovaResults = analysis['anova']
        bestFeature = min(anovaResults.items(), key=lambda x: x[1]['p_value'])
        
        insights.append({
            'tipo': 'diferenciacao',
            'titulo': 'Feature Mais Discriminante',
            'descricao': f"'{bestFeature[0]}' melhor diferencia os clusters (p < {bestFeature[1]['p_value']:.4f})",
            'acao': f"Foque no desenvolvimento de '{bestFeature[0]}' para progressão entre níveis"
        })
        
        # Insight 3: Correlação forte
        strongCorr = analysis['correlations']['pares_significativos']
        if strongCorr:
            strongest = max(strongCorr, key=lambda x: abs(x['correlacao']))
            
            insights.append({
                'tipo': 'correlacao',
                'titulo': 'Correlação Forte Detectada',
                'descricao': f"{strongest['var1']} e {strongest['var2']} têm correlação de {strongest['correlacao']:.2f}",
                'acao': 'Treine estas capacidades em conjunto para melhor resultado'
            })
        
        # Insight 4: Qualidade do clustering
        quality = analysis['cluster_analysis']['qualidade']
        
        if quality['silhouette'] > 0.5:
            qualityLevel = 'excelente'
        elif quality['silhouette'] > 0.3:
            qualityLevel = 'boa'
        else:
            qualityLevel = 'moderada'
        
        insights.append({
            'tipo': 'clustering',
            'titulo': 'Qualidade da Classificação',
            'descricao': f"A separação entre clusters é {qualityLevel} (silhouette: {quality['silhouette']:.2f})",
            'acao': 'Os clusters refletem diferenças reais entre níveis de atletas' if qualityLevel != 'moderada' 
                   else 'Considere revisar os critérios de clustering'
        })
        
        # Insight 5: Desempenho por sexo
        malePerf = df[df['sexo'] == 'M']['score_total'].mean()
        femalePerf = df[df['sexo'] == 'F']['score_total'].mean()
        
        if abs(malePerf - femalePerf) > 10:
            insights.append({
                'tipo': 'sexo',
                'titulo': 'Diferença de Desempenho por Sexo',
                'descricao': f"Diferença de {abs(malePerf - femalePerf):.1f} pontos entre sexos",
                'acao': 'Considere programas de treinamento específicos por sexo'
            })
        
        data['analysis']['insights'] = insights
        return data
    
    
    def assembleAnalysisReport(self, data: Dict) -> Dict:
        """Monta relatório final de análise"""
        
        analysis = data['analysis']
        df = data['df']
        
        report = {
            'resumo_executivo': {
                'total_atletas': len(df),
                'clusters': len(df['cluster_id'].unique()),
                'features_analisadas': len(self.features),
                'qualidade_clustering': analysis['cluster_analysis']['qualidade']['silhouette']
            },
            'analise_descritiva': analysis['descriptive'],
            'correlacoes': analysis['correlations'],
            'analise_clusters': analysis['cluster_analysis'],
            'anova': analysis['anova'],
            'regressoes': analysis['regressions'],
            'insights': analysis['insights']
        }
        
        return report
    
    
    def _interpretCorrelation(self, corr: float) -> str:
        """Interpreta valor de correlação"""
        
        absCorr = abs(corr)
        
        if absCorr > 0.8:
            strength = "muito forte"
        elif absCorr > 0.6:
            strength = "forte"
        elif absCorr > 0.4:
            strength = "moderada"
        else:
            strength = "fraca"
        
        direction = "positiva" if corr > 0 else "negativa"
        
        return f"Correlação {strength} {direction}"
    
    
    def _interpretClusterQuality(self, silhouette: float, daviesBouldin: float) -> str:
        """Interpreta métricas de qualidade do clustering"""
        
        if silhouette > 0.5 and daviesBouldin < 1.0:
            return "Excelente separação entre clusters. Os grupos são bem definidos."
        elif silhouette > 0.3:
            return "Boa separação entre clusters. Os grupos têm características distintas."
        else:
            return "Separação moderada. Considere revisar os critérios de agrupamento."
    
    
    def _interpretANOVA(self, pValue: float) -> str:
        """Interpreta resultado do ANOVA"""
        
        if pValue < 0.001:
            return "Diferença altamente significativa entre clusters (p < 0.001)"
        elif pValue < 0.05:
            return "Diferença significativa entre clusters (p < 0.05)"
        else:
            return "Sem diferença significativa entre clusters (p ≥ 0.05)"