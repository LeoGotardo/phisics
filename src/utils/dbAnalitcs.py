import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Tuple, Dict, List, Literal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import f_oneway

from src.model.athleteModel import Athlete
from src.config import Config


class DBAnalytics:
    """
    Classe para análises estatísticas avançadas do banco de dados.
    Gera visualizações e métricas sobre os atletas cadastrados.
    """
    
    def __init__(self):
        self.session = Config.session
        
        # Configurar estilo dos gráficos
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
        self.cluster_names = {
            0: 'Iniciante',
            1: 'Intermediário', 
            2: 'Competitivo',
            3: 'Elite'
        }
        
        self.cluster_colors = {
            'Elite': '#FF6B6B',
            'Competitivo': '#4ECDC4',
            'Intermediário': '#FFE66D',
            'Iniciante': '#95E1D3'
        }
    
    
    def getAthletes(self) -> Tuple[bool, pd.DataFrame | str]:
        """
        Busca todos os atletas do banco e retorna como DataFrame.
        
        Returns:
            Tupla (sucesso, DataFrame ou mensagem de erro)
        """
        try:
            with Config.app.app_context():
                athletes = self.session.query(Athlete).all()
            
            if len(athletes) == 0:
                return False, "Nenhum atleta cadastrado no banco de dados"
            
            # Converter para DataFrame
            data = [athlete.dict() for athlete in athletes]
            df = pd.DataFrame(data)
            
            # Preparar dados
            df['sexo_encoded'] = df['sexo'].map({'M': 1, 'F': 0})
            
            return True, df
            
        except Exception as e:
            return False, f"Erro ao buscar atletas: {str(e)}"
    
    
    def analyzeCorrelation(self, df: pd.DataFrame) -> Dict:
        """
        Analisa correlação entre potência superior e inferior.
        
        Args:
            df: DataFrame com dados dos atletas
            
        Returns:
            Dicionário com resultados da análise
        """
        try:
            # Criar variáveis de potência
            potencia_superior = df['arremesso'].values
            potencia_inferior = df['saltoHorizontal'].values
            
            # Calcular correlação
            r_value, p_value = stats.pearsonr(potencia_superior, potencia_inferior)
            
            # Interpretação
            if p_value < 0.001:
                significancia = "p < 0.001 - Altamente significativo"
            elif p_value < 0.05:
                significancia = f"p = {p_value:.3f} - Significativo"
            else:
                significancia = f"p = {p_value:.3f} - Não significativo"
            
            if abs(r_value) > 0.7:
                interpretacao = "Correlação forte positiva"
            elif abs(r_value) > 0.4:
                interpretacao = "Correlação moderada positiva"
            else:
                interpretacao = "Correlação fraca"
            
            return {
                'r_value': float(r_value),
                'p_value': float(p_value),
                'significancia': significancia,
                'interpretacao': interpretacao,
                'potencia_superior': potencia_superior.tolist(),
                'potencia_inferior': potencia_inferior.tolist()
            }
            
        except Exception as e:
            return {'erro': str(e)}
    
    
    def analyzeCoreStrength(self, df: pd.DataFrame) -> Dict:
        """
        Analisa força do core por cluster usando ANOVA.
        
        Args:
            df: DataFrame com dados dos atletas
            
        Returns:
            Dicionário com resultados da análise
        """
        try:
            results = {}
            
            # Estatísticas por cluster
            for cluster_id, nivel in self.cluster_names.items():
                dados = df[df['cluster'] == cluster_id]['abdominais']
                
                if len(dados) > 0:
                    results[nivel] = {
                        'media': float(dados.mean()),
                        'desvio': float(dados.std()),
                        'min': int(dados.min()),
                        'max': int(dados.max()),
                        'count': int(len(dados))
                    }
            
            # ANOVA - Teste de diferença entre grupos
            grupos = [
                df[df['cluster'] == i]['abdominais'].values 
                for i in range(4)
                if len(df[df['cluster'] == i]) > 0
            ]
            
            if len(grupos) >= 2:
                f_stat, p_anova = f_oneway(*grupos)
                
                results['anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_anova),
                    'significativo': p_anova < 0.001
                }
            
            return results
            
        except Exception as e:
            return {'erro': str(e)}
    
    
    def analyzeClusterProfile(self, df: pd.DataFrame) -> Dict:
        """
        Analisa perfil médio de cada cluster (normalizado 0-100).
        
        Args:
            df: DataFrame com dados dos atletas
            
        Returns:
            Dicionário com perfis dos clusters
        """
        try:
            features = ['arremesso', 'saltoHorizontal', 'abdominais']
            df_normalized = df.copy()
            
            # Normalizar features para 0-100
            for feat in features:
                min_val = df[feat].min()
                max_val = df[feat].max()
                df_normalized[f'{feat}_norm'] = (
                    ((df[feat] - min_val) / (max_val - min_val)) * 100
                )
            
            results = {}
            
            for cluster_id, nivel in self.cluster_names.items():
                dados_nivel = df_normalized[df_normalized['cluster'] == cluster_id]
                
                if len(dados_nivel) > 0:
                    scores = []
                    for feat in features:
                        score = dados_nivel[f'{feat}_norm'].mean()
                        scores.append(score)
                    
                    # Coeficiente de variação (homogeneidade)
                    cv = (
                        dados_nivel[[f'{f}_norm' for f in features]].std().mean() / 
                        dados_nivel[[f'{f}_norm' for f in features]].mean().mean()
                    )
                    
                    results[nivel] = {
                        'scores': [float(s) for s in scores],
                        'min_score': float(min(scores)),
                        'max_score': float(max(scores)),
                        'coef_variacao': float(cv),
                        'homogeneo': cv < 0.2
                    }
            
            return results
            
        except Exception as e:
            return {'erro': str(e)}
    
    
    def generateVisualization(self, df: pd.DataFrame) -> Tuple[bool, bytes | str]:
        """
        Gera visualização completa com todos os gráficos.
        
        Args:
            df: DataFrame com dados dos atletas
            
        Returns:
            Tupla (sucesso, bytes da imagem ou mensagem de erro)
        """
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Correlação Potência
            ax_corr = fig.add_subplot(gs[0, 2])
            self._plotCorrelation(ax_corr, df)
            
            # 2. Força do Core
            ax_core = fig.add_subplot(gs[1, 0])
            self._plotCoreStrength(ax_core, df)
            
            # 3. Perfil Médio (Barras)
            ax_profile = fig.add_subplot(gs[1, 1])
            self._plotProfile(ax_profile, df)
            
            # 4. Scatter Arremesso vs Salto
            ax_scatter = fig.add_subplot(gs[2, 0])
            self._plotScatterPerformance(ax_scatter, df)
            
            # 5. Distribuição de Sexo por Cluster
            ax_sex = fig.add_subplot(gs[2, 1])
            self._plotSexDistribution(ax_sex, df)
            
            # Converter figura para bytes
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return True, buf.read()
            
        except Exception as e:
            return False, f"Erro ao gerar visualização: {str(e)}"
    
    
    def _plotCorrelation(self, ax, df: pd.DataFrame):
        """Gráfico de correlação entre potência superior e inferior."""
        ax.scatter(df['arremesso'], df['saltoHorizontal'], 
                  alpha=0.5, c=df['cluster'], cmap='viridis')
        ax.set_xlabel('Potência Superior (Arremesso)', fontsize=10)
        ax.set_ylabel('Potência Inferior (Salto)', fontsize=10)
        
        # Adicionar linha de tendência
        z = np.polyfit(df['arremesso'], df['saltoHorizontal'], 1)
        p = np.poly1d(z)
        ax.plot(df['arremesso'], p(df['arremesso']), 
               "r--", alpha=0.8, linewidth=2)
        
        # Calcular correlação
        r_value, _ = stats.pearsonr(df['arremesso'], df['saltoHorizontal'])
        ax.set_title(f'Correlação: r={r_value:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    
    def _plotCoreStrength(self, ax, df: pd.DataFrame):
        """Box plot da força do core por cluster."""
        data_core = [
            df[df['cluster'] == i]['abdominais'].values 
            for i in range(4)
            if len(df[df['cluster'] == i]) > 0
        ]
        
        labels = [
            self.cluster_names[i] 
            for i in range(4)
            if len(df[df['cluster'] == i]) > 0
        ]
        
        bp = ax.boxplot(data_core, labels=labels, patch_artist=True)
        
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(self.cluster_colors[label])
        
        ax.set_ylabel('Abdominais (rep/min)', fontsize=10)
        ax.set_title('Força do Core por Cluster', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    
    def _plotProfile(self, ax, df: pd.DataFrame):
        """Gráfico de perfil médio por cluster."""
        features = ['arremesso', 'saltoHorizontal', 'abdominais']
        df_normalized = df.copy()
        
        for feat in features:
            min_val = df[feat].min()
            max_val = df[feat].max()
            df_normalized[f'{feat}_norm'] = (
                ((df[feat] - min_val) / (max_val - min_val)) * 100
            )
        
        niveis = [
            self.cluster_names[i] 
            for i in range(4)
            if len(df[df['cluster'] == i]) > 0
        ]
        
        scores_medios = []
        for i in range(4):
            if len(df[df['cluster'] == i]) > 0:
                dados_nivel = df_normalized[df_normalized['cluster'] == i]
                score_medio = dados_nivel[
                    [f'{f}_norm' for f in features]
                ].mean().mean()
                scores_medios.append(score_medio)
        
        colors = [self.cluster_colors[nivel] for nivel in niveis]
        bars = ax.barh(niveis, scores_medios, color=colors)
        
        ax.set_xlabel('Score Normalizado (0-100)', fontsize=10)
        ax.set_title('Perfil Médio por Cluster', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        
        for i, v in enumerate(scores_medios):
            ax.text(v + 2, i, f'{v:.0f}', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
    
    
    def _plotScatterPerformance(self, ax, df: pd.DataFrame):
        """Scatter plot: Arremesso vs Salto colorido por cluster."""
        for i in range(4):
            if len(df[df['cluster'] == i]) > 0:
                df_cluster = df[df['cluster'] == i]
                nivel = self.cluster_names[i]
                
                ax.scatter(df_cluster['arremesso'], df_cluster['saltoHorizontal'],
                          c=self.cluster_colors[nivel], label=nivel, 
                          alpha=0.6, s=100, edgecolors='black')
        
        ax.set_xlabel('Arremesso (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Salto Horizontal (m)', fontsize=12, fontweight='bold')
        ax.set_title('Relação: Potência Superior vs Inferior', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Cluster', loc='best')
        ax.grid(True, alpha=0.3)
    
    
    def _plotSexDistribution(self, ax, df: pd.DataFrame):
        """Gráfico de distribuição de sexo por cluster."""
        masculino = [
            len(df[(df['cluster'] == i) & (df['sexo'] == 'M')]) 
            for i in range(4)
            if len(df[df['cluster'] == i]) > 0
        ]
        
        feminino = [
            len(df[(df['cluster'] == i) & (df['sexo'] == 'F')]) 
            for i in range(4)
            if len(df[df['cluster'] == i]) > 0
        ]
        
        niveis = [
            self.cluster_names[i] 
            for i in range(4)
            if len(df[df['cluster'] == i]) > 0
        ]
        
        x = range(len(niveis))
        
        bars1 = ax.bar(x, masculino, label='Masculino', 
                      color='#4a9eff', alpha=0.8)
        bars2 = ax.bar(x, feminino, bottom=masculino, label='Feminino',
                      color='#ec4899', alpha=0.8)
        
        # Adicionar valores
        for i, (m, f) in enumerate(zip(masculino, feminino)):
            if m > 0:
                ax.text(i, m/2, str(m), ha='center', va='center',
                       fontweight='bold', color='white')
            if f > 0:
                ax.text(i, m + f/2, str(f), ha='center', va='center',
                       fontweight='bold', color='white')
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Atletas', fontsize=12, fontweight='bold')
        ax.set_title('Distribuição de Sexo por Cluster', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(niveis)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    
    def generateFullReport(self) -> Tuple[bool, Dict | str]:
        """
        Gera relatório completo com todas as análises.
        
        Returns:
            Tupla (sucesso, dicionário com análises ou mensagem de erro)
        """
        try:
            # Buscar dados
            success, df_or_error = self.getAthletes()
            if not success:
                return False, df_or_error
            
            df = df_or_error
            
            # Executar análises
            correlation = self.analyzeCorrelation(df)
            core_strength = self.analyzeCoreStrength(df)
            cluster_profile = self.analyzeClusterProfile(df)
            
            # Gerar visualização
            viz_success, viz_data = self.generateVisualization(df)
            
            # Montar relatório
            report = {
                'total_atletas': len(df),
                'distribuicao': df['cluster'].value_counts().to_dict(),
                'correlacao': correlation,
                'forca_core': core_strength,
                'perfil_clusters': cluster_profile,
                'visualizacao_disponivel': viz_success
            }
            
            if viz_success:
                report['visualizacao_bytes'] = viz_data
            
            return True, report
            
        except Exception as e:
            return False, f"Erro ao gerar relatório: {str(e)}"
    
    
    def saveVisualization(self, filepath: str = 'analise_estatistica.png') -> Tuple[bool, str]:
        """
        Salva visualização em arquivo.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            success, df_or_error = self.getAthletes()
            if not success:
                return False, df_or_error
            
            viz_success, viz_data = self.generateVisualization(df_or_error)
            
            if not viz_success:
                return False, viz_data
            
            with open(filepath, 'wb') as f:
                f.write(viz_data)
            
            return True, f"Visualização salva em {filepath}"
            
        except Exception as e:
            return False, f"Erro ao salvar visualização: {str(e)}"