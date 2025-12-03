import pandas as pd

from src.model.athleteModel import Athlete
from src.utils.dataclasses import Elo
from sqlalchemy import func, and_
from src.config import Config
from datetime import datetime
from typing import Dict


class DBAnalyticsElo(Elo):
    """
    Pipeline para análises diretas no banco de dados.
    Queries otimizadas para estatísticas agregadas e tendências temporais.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self.clusterNames = {
            0: 'Iniciante',
            1: 'Intermediário',
            2: 'Competitivo',
            3: 'Elite'
        }
        
        self._buildChain()
    
    
    def _buildChain(self) -> None:
        """Constrói a cadeia de análises no banco"""
        
        self.chain = [
            self.getBasicStatistics,
            self.getClusterDistribution,
            self.getGenderAnalysis,
            self.getAgeDistribution,
            self.getPerformanceRankings,
            self.getAnomalies,
            self.getTrends,
            self.assembleDBReport
        ]
    
    
    def getBasicStatistics(self, _: None = None) -> Dict:
        """Estatísticas básicas agregadas do banco"""
        
        with Config.app.app_context():
            session = Config.session
            
            stats = {
                'total_atletas': session.query(Athlete).count(),
                'por_sexo': {
                    'masculino': session.query(Athlete).filter(Athlete.sexo == 'M').count(),
                    'feminino': session.query(Athlete).filter(Athlete.sexo == 'F').count()
                },
                'medias_gerais': {
                    'altura': float(session.query(func.avg(Athlete.altura)).scalar() or 0),
                    'envergadura': float(session.query(func.avg(Athlete.envergadura)).scalar() or 0),
                    'arremesso': float(session.query(func.avg(Athlete.arremesso)).scalar() or 0),
                    'saltoHorizontal': float(session.query(func.avg(Athlete.saltoHorizontal)).scalar() or 0),
                    'abdominais': float(session.query(func.avg(Athlete.abdominais)).scalar() or 0)
                },
                'maximos': {
                    'altura': float(session.query(func.max(Athlete.altura)).scalar() or 0),
                    'envergadura': float(session.query(func.max(Athlete.envergadura)).scalar() or 0),
                    'arremesso': float(session.query(func.max(Athlete.arremesso)).scalar() or 0),
                    'saltoHorizontal': float(session.query(func.max(Athlete.saltoHorizontal)).scalar() or 0),
                    'abdominais': float(session.query(func.max(Athlete.abdominais)).scalar() or 0)
                },
                'minimos': {
                    'altura': float(session.query(func.min(Athlete.altura)).scalar() or 0),
                    'envergadura': float(session.query(func.min(Athlete.envergadura)).scalar() or 0),
                    'arremesso': float(session.query(func.min(Athlete.arremesso)).scalar() or 0),
                    'saltoHorizontal': float(session.query(func.min(Athlete.saltoHorizontal)).scalar() or 0),
                    'abdominais': float(session.query(func.min(Athlete.abdominais)).scalar() or 0)
                }
            }
        
        return {'db_stats': stats, 'analytics': {}}
    
    
    def getClusterDistribution(self, data: Dict) -> Dict:
        """Distribuição detalhada por cluster"""
        
        with Config.app.app_context():
            session = Config.session
            
            distribution = {}
            
            for clusterId, clusterName in self.clusterNames.items():
                count = session.query(Athlete).filter(Athlete.cluster == clusterId).count()
                
                if count > 0:
                    # Médias por cluster
                    avgStats = session.query(
                        func.avg(Athlete.altura),
                        func.avg(Athlete.envergadura),
                        func.avg(Athlete.arremesso),
                        func.avg(Athlete.saltoHorizontal),
                        func.avg(Athlete.abdominais)
                    ).filter(Athlete.cluster == clusterId).first()
                    
                    # Contagem por sexo
                    maleCount = session.query(Athlete).filter(
                        and_(Athlete.cluster == clusterId, Athlete.sexo == 'M')
                    ).count()
                    
                    femaleCount = session.query(Athlete).filter(
                        and_(Athlete.cluster == clusterId, Athlete.sexo == 'F')
                    ).count()
                    
                    distribution[clusterName] = {
                        'total': count,
                        'percentual': round((count / data['db_stats']['total_atletas']) * 100, 2),
                        'sexo': {
                            'masculino': maleCount,
                            'feminino': femaleCount
                        },
                        'medias': {
                            'altura': round(float(avgStats[0] or 0), 2),
                            'envergadura': round(float(avgStats[1] or 0), 2),
                            'arremesso': round(float(avgStats[2] or 0), 2),
                            'saltoHorizontal': round(float(avgStats[3] or 0), 2),
                            'abdominais': round(float(avgStats[4] or 0), 2)
                        }
                    }
        
        data['analytics']['distribuicao_clusters'] = distribution
        return data
    
    
    def getGenderAnalysis(self, data: Dict) -> Dict:
        """Análise comparativa por gênero"""
        
        with Config.app.app_context():
            session = Config.session
            
            genderAnalysis = {}
            
            for gender, label in [('M', 'masculino'), ('F', 'feminino')]:
                athletes = session.query(Athlete).filter(Athlete.sexo == gender).all()
                
                if athletes:
                    df = pd.DataFrame([a.dict() for a in athletes])
                    
                    genderAnalysis[label] = {
                        'total': len(athletes),
                        'medias': {
                            'altura': float(df['altura'].mean()),
                            'envergadura': float(df['envergadura'].mean()),
                            'arremesso': float(df['arremesso'].mean()),
                            'saltoHorizontal': float(df['saltoHorizontal'].mean()),
                            'abdominais': float(df['abdominais'].mean())
                        },
                        'desvios': {
                            'altura': float(df['altura'].std()),
                            'envergadura': float(df['envergadura'].std()),
                            'arremesso': float(df['arremesso'].std()),
                            'saltoHorizontal': float(df['saltoHorizontal'].std()),
                            'abdominais': float(df['abdominais'].std())
                        },
                        'distribuicao_clusters': df['cluster'].value_counts().to_dict()
                    }
        
        data['analytics']['analise_genero'] = genderAnalysis
        return data
    
    
    def getAgeDistribution(self, data: Dict) -> Dict:
        """Distribuição por faixa etária"""
        
        with Config.app.app_context():
            session = Config.session
            
            athletes = session.query(Athlete).all()
            
            if athletes:
                df = pd.DataFrame([a.dict() for a in athletes])
                
                # Calcular idade
                today = pd.to_datetime('today')
                df['idade'] = (today - pd.to_datetime(df['dataNascimento'])).dt.days // 365
                
                # Criar faixas etárias
                bins = [0, 15, 20, 25, 30, 35, 100]
                labels = ['<15', '15-19', '20-24', '25-29', '30-34', '35+']
                df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels)
                
                ageDistribution = {
                    'por_faixa': df['faixa_etaria'].value_counts().to_dict(),
                    'idade_media': float(df['idade'].mean()),
                    'idade_mediana': float(df['idade'].median()),
                    'idade_minima': int(df['idade'].min()),
                    'idade_maxima': int(df['idade'].max())
                }
                
                # Performance por faixa etária
                performanceByAge = {}
                
                for faixa in labels:
                    dfFaixa = df[df['faixa_etaria'] == faixa]
                    
                    if len(dfFaixa) > 0:
                        performanceByAge[faixa] = {
                            'total': len(dfFaixa),
                            'arremesso_medio': float(dfFaixa['arremesso'].mean()),
                            'salto_medio': float(dfFaixa['saltoHorizontal'].mean()),
                            'abdominais_medio': float(dfFaixa['abdominais'].mean())
                        }
                
                ageDistribution['performance_por_faixa'] = performanceByAge
                
                data['analytics']['distribuicao_idade'] = ageDistribution
        
        return data
    
    
    def getPerformanceRankings(self, data: Dict) -> Dict:
        """Rankings e recordes de performance"""
        
        with Config.app.app_context():
            session = Config.session
            
            rankings = {
                'top_arremesso': [],
                'top_salto': [],
                'top_abdominais': [],
                'top_geral': []
            }
            
            # Top 10 Arremesso
            topArremesso = session.query(Athlete).order_by(
                Athlete.arremesso.desc()
            ).limit(10).all()
            
            rankings['top_arremesso'] = [
                {
                    'nome': a.nome,
                    'valor': float(a.arremesso),
                    'cluster': self.clusterNames.get(a.cluster, 'N/A')
                }
                for a in topArremesso
            ]
            
            # Top 10 Salto
            topSalto = session.query(Athlete).order_by(
                Athlete.saltoHorizontal.desc()
            ).limit(10).all()
            
            rankings['top_salto'] = [
                {
                    'nome': a.nome,
                    'valor': float(a.saltoHorizontal),
                    'cluster': self.clusterNames.get(a.cluster, 'N/A')
                }
                for a in topSalto
            ]
            
            # Top 10 Abdominais
            topAbdominais = session.query(Athlete).order_by(
                Athlete.abdominais.desc()
            ).limit(10).all()
            
            rankings['top_abdominais'] = [
                {
                    'nome': a.nome,
                    'valor': int(a.abdominais),
                    'cluster': self.clusterNames.get(a.cluster, 'N/A')
                }
                for a in topAbdominais
            ]
            
            # Top 10 Geral (score composto)
            allAthletes = session.query(Athlete).all()
            
            if allAthletes:
                df = pd.DataFrame([a.dict() for a in allAthletes])
                df['score'] = (
                    df['arremesso'] * 0.3 + 
                    df['saltoHorizontal'] * 0.3 + 
                    df['abdominais'] * 0.2 +
                    df['altura'] * 0.1 +
                    df['envergadura'] * 0.1
                )
                
                topGeral = df.nlargest(10, 'score')
                
                rankings['top_geral'] = [
                    {
                        'nome': row['nome'],
                        'score': float(row['score']),
                        'cluster': self.clusterNames.get(row['cluster'], 'N/A')
                    }
                    for _, row in topGeral.iterrows()
                ]
        
        data['analytics']['rankings'] = rankings
        return data
    
    
    def getAnomalies(self, data: Dict) -> Dict:
        """Detecta anomalias e outliers"""
        
        with Config.app.app_context():
            session = Config.session
            athletes = session.query(Athlete).all()
            
            if not athletes:
                return data
            
            df = pd.DataFrame([a.dict() for a in athletes])
            
            anomalies = {
                'outliers': [],
                'valores_extremos': [],
                'inconsistencias': []
            }
            
            features = ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']
            
            # Detectar outliers usando IQR
            for feature in features:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lowerBound = Q1 - 1.5 * IQR
                upperBound = Q3 + 1.5 * IQR
                
                outliers = df[(df[feature] < lowerBound) | (df[feature] > upperBound)]
                
                for _, row in outliers.iterrows():
                    anomalies['outliers'].append({
                        'atleta': row['nome'],
                        'feature': feature,
                        'valor': float(row[feature]),
                        'limites': {'inferior': float(lowerBound), 'superior': float(upperBound)}
                    })
            
            # Detectar inconsistências (envergadura < altura)
            inconsistent = df[df['envergadura'] < df['altura']]
            
            for _, row in inconsistent.iterrows():
                anomalies['inconsistencias'].append({
                    'atleta': row['nome'],
                    'tipo': 'envergadura_menor_altura',
                    'altura': float(row['altura']),
                    'envergadura': float(row['envergadura'])
                })
            
            data['analytics']['anomalias'] = anomalies
        
        return data
    
    
    def getTrends(self, data: Dict) -> Dict:
        """Identifica tendências nos dados"""
        
        with Config.app.app_context():
            session = Config.session
            athletes = session.query(Athlete).all()
            
            if not athletes:
                return data
            
            df = pd.DataFrame([a.dict() for a in athletes])
            
            # Calcular idade
            today = pd.to_datetime('today')
            df['idade'] = (today - pd.to_datetime(df['dataNascimento'])).dt.days // 365
            
            trends = {
                'crescimento_performance': {},
                'equilibrio_desenvolvimento': {},
                'potencial_progressao': []
            }
            
            # Análise de crescimento por idade
            for clusterId, clusterName in self.clusterNames.items():
                dfCluster = df[df['cluster'] == clusterId]
                
                if len(dfCluster) > 0:
                    trends['crescimento_performance'][clusterName] = {
                        'idade_media': float(dfCluster['idade'].mean()),
                        'performance_media': float(
                            dfCluster['arremesso'].mean() + 
                            dfCluster['saltoHorizontal'].mean()
                        )
                    }
            
            # Equilíbrio entre capacidades
            df['ratio_superior_inferior'] = df['arremesso'] / (df['saltoHorizontal'] + 0.1)
            
            equilibrado = df[
                (df['ratio_superior_inferior'] > 0.8) & 
                (df['ratio_superior_inferior'] < 1.2)
            ]
            
            trends['equilibrio_desenvolvimento'] = {
                'atletas_equilibrados': len(equilibrado),
                'percentual': round((len(equilibrado) / len(df)) * 100, 2),
                'exemplos': equilibrado.head(5)['nome'].tolist()
            }
            
            # Potencial de progressão (iniciantes/intermediários com boas métricas)
            potencial = df[
                (df['cluster'].isin([0, 1])) &
                (df['arremesso'] > df['arremesso'].quantile(0.7))
            ]
            
            trends['potencial_progressao'] = [
                {
                    'nome': row['nome'],
                    'cluster_atual': self.clusterNames.get(row['cluster'], 'N/A'),
                    'score': float(
                        row['arremesso'] * 0.4 + 
                        row['saltoHorizontal'] * 0.4 + 
                        row['abdominais'] * 0.2
                    )
                }
                for _, row in potencial.head(10).iterrows()
            ]
            
            data['analytics']['tendencias'] = trends
        
        return data
    
    
    def assembleDBReport(self, data: Dict) -> Dict:
        """Monta relatório final de analytics do banco"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'estatisticas_basicas': data['db_stats'],
            'analytics': data['analytics'],
            'resumo': {
                'total_atletas': data['db_stats']['total_atletas'],
                'clusters_ativos': len(data['analytics']['distribuicao_clusters']),
                'anomalias_detectadas': len(data['analytics'].get('anomalias', {}).get('outliers', [])),
                'atletas_com_potencial': len(
                    data['analytics'].get('tendencias', {}).get('potencial_progressao', [])
                )
            }
        }
        
        return report