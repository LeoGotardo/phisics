import pandas as pd, sys, zipfile, io, matplotlib.pyplot as plt, seaborn as sns

from src.utils.dataclasses import Column, Elo
from src.model.athleteModel import Athlete
from src.model.knnModel import KNNModel
from src.config import Config
from typing import Literal


class CSVExportElo(Elo):
    """
    Pipeline para exportação de dados CSV com cada coluna validada.
    Herda de Elo e implementa a cadeia de processamento.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self.COLUMNS: list[Column] = Config.COLUMNS
        self.knnModel = KNNModel()
        
        # Configurar estilo dos gráficos
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
        self._buildChain()
    
    
    def _buildChain(self) -> None:
        """Constrói a cadeia de funções para gerar o CSV"""
        
        self.chain = [
            self.getAthletes,
            self.convertAthletesData,
            self.generateCSV
        ]
    
    
    def getAthletes(self, athletesIds: list[str] = None) -> list[Athlete]:
        """
        Busca atletas do banco de dados.
        
        Args:
            athletesIds: Lista de IDs específicos (opcional)
            
        Returns:
            Lista de objetos Athlete
        """
        
        with Config.app.app_context():
            if athletesIds:
                athletes = Config.session.query(Athlete).filter(
                    Athlete.id.in_(athletesIds)
                ).all()
            else:
                athletes = Config.session.query(Athlete).all()
        
        return athletes
    
    
    def convertAthletesData(self, athletes: list[Athlete]) -> pd.DataFrame:
        """
        Converte a lista de atletas em DataFrame.
        
        Args:
            athletes: Lista de objetos Athlete
            
        Returns:
            DataFrame com os dados dos atletas
        """
        
        data = [athlete.dict() for athlete in athletes]
        return pd.DataFrame(data)
    
    
    def generateCSV(self, dataframe: pd.DataFrame) -> bytes:
        """
        Gera arquivo CSV a partir do DataFrame.
        
        Args:
            dataframe: DataFrame com dados
            
        Returns:
            CSV em bytes
        """
        
        return dataframe.to_csv(index=False).encode('utf-8')
    
    
    def getKNNMetrics(self, dataframe: pd.DataFrame) -> dict:
        """
        Retorna as métricas do KNN.

        Args:
            dataframe: DataFrame com dados dos atletas

        Returns:
            dict: Métricas do KNN
        """
        
        try:
            # Tentar obter métricas do modelo já treinado
            metrics = self.knnModel.getModelMetrics(dataframe)
            return metrics
        except Exception as e:
            print(f"Erro ao obter métricas KNN: {e}")
            return {}
    
    
    def generateGraphs(self, dataframe: pd.DataFrame) -> dict[str, bytes]:
        """
        Gera todos os gráficos estatísticos.
        
        Args:
            dataframe: DataFrame com dados dos atletas
            
        Returns:
            Dicionário com nome_arquivo: bytes_imagem
        """
        
        graphs = {}
        
        graphs['01_distribuicao_clusters.png'] = self._graphDistribuicaoClusters(dataframe)
        graphs['02_boxplot_features.png'] = self._graphBoxplotFeatures(dataframe)
        graphs['03_correlacao_features.png'] = self._graphCorrelacao(dataframe)
        graphs['04_altura_cluster.png'] = self._graphAlturaPorCluster(dataframe)
        graphs['05_arremesso_vs_salto.png'] = self._graphArremessoVsSalto(dataframe)
        graphs['06_sexo_cluster.png'] = self._graphSexoPorCluster(dataframe)
        
        return graphs
    
    
    def _graphDistribuicaoClusters(self, df: pd.DataFrame) -> bytes:
        """Gráfico de barras com distribuição de atletas por cluster"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        clusterCounts = df['cluster'].value_counts().sort_index()
        clusterNames = ['Iniciante', 'Intermediário', 'Competitivo', 'Elite']
        colors = ['#6b7280', '#10b981', '#7c3aed', '#4a9eff']
        
        bars = ax.bar(clusterNames, clusterCounts.values, color=colors, alpha=0.8, edgecolor='black')
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Atletas', fontsize=12, fontweight='bold')
        ax.set_title('Distribuição de Atletas por Cluster', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        return self._figToBytes(fig)
    
    
    def _graphBoxplotFeatures(self, df: pd.DataFrame) -> bytes:
        """Box plots das principais features por cluster"""
        
        features = ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']
        featureLabels = ['Altura (cm)', 'Envergadura (cm)', 'Arremesso (m)', 
                        'Salto Horizontal (m)', 'Abdominais (rep/min)']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        clusterNames = ['Iniciante', 'Intermediário', 'Competitivo', 'Elite']
        colors = ['#6b7280', '#10b981', '#7c3aed', '#4a9eff']
        
        for idx, (feature, label) in enumerate(zip(features, featureLabels)):
            ax = axes[idx]
            
            # Preparar dados por cluster
            dataByCluster = [df[df['cluster'] == i][feature].dropna() for i in range(4)]
            
            bp = ax.boxplot(dataByCluster, labels=clusterNames, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_ylabel('Valor', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        fig.delaxes(axes[-1])
        
        plt.suptitle('Distribuição das Features por Cluster', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        return self._figToBytes(fig)
    
    
    def _graphCorrelacao(self, df: pd.DataFrame) -> bytes:
        """Matriz de correlação entre features numéricas"""
        
        numericCols = ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']
        corrMatrix = df[numericCols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corrMatrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Matriz de Correlação entre Features', fontsize=14, fontweight='bold', pad=20)
        
        labels = ['Altura', 'Envergadura', 'Arremesso', 'Salto H.', 'Abdominais']
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        return self._figToBytes(fig)
    
    
    def _graphAlturaPorCluster(self, df: pd.DataFrame) -> bytes:
        """Distribuição de altura por cluster e sexo"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        clusterNames = ['Iniciante', 'Intermediário', 'Competitivo', 'Elite']
        
        dfMasculino = df[df['sexo'] == 'M']
        dfFeminino = df[df['sexo'] == 'F']
        
        x = range(4)
        width = 0.35
        
        alturaMasculino = [dfMasculino[dfMasculino['cluster'] == i]['altura'].mean() 
                          for i in range(4)]
        alturaFeminino = [dfFeminino[dfFeminino['cluster'] == i]['altura'].mean() 
                         for i in range(4)]
        
        bars1 = ax.bar([i - width/2 for i in x], alturaMasculino, width, 
                      label='Masculino', color='#4a9eff', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], alturaFeminino, width, 
                      label='Feminino', color='#ec4899', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not pd.isna(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Altura Média (cm)', fontsize=12, fontweight='bold')
        ax.set_title('Altura Média por Cluster e Sexo', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(clusterNames)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        return self._figToBytes(fig)
    
    
    def _graphArremessoVsSalto(self, df: pd.DataFrame) -> bytes:
        """Scatter plot: Arremesso vs Salto colorido por cluster"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        clusterNames = ['Iniciante', 'Intermediário', 'Competitivo', 'Elite']
        colors = ['#6b7280', '#10b981', '#7c3aed', '#4a9eff']
        
        for i, (cluster, color) in enumerate(zip(clusterNames, colors)):
            dfCluster = df[df['cluster'] == i]
            ax.scatter(dfCluster['arremesso'], dfCluster['saltoHorizontal'], 
                      c=color, label=cluster, alpha=0.6, s=100, edgecolors='black')
        
        ax.set_xlabel('Arremesso (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Salto Horizontal (m)', fontsize=12, fontweight='bold')
        ax.set_title('Relação: Potência Superior vs Inferior', fontsize=14, fontweight='bold')
        ax.legend(title='Cluster', loc='best')
        ax.grid(True, alpha=0.3)
        
        return self._figToBytes(fig)
    
    
    def _graphSexoPorCluster(self, df: pd.DataFrame) -> bytes:
        """Gráfico de barras empilhadas: distribuição de sexo por cluster"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        clusterNames = ['Iniciante', 'Intermediário', 'Competitivo', 'Elite']
        
        masculino = [len(df[(df['cluster'] == i) & (df['sexo'] == 'M')]) for i in range(4)]
        feminino = [len(df[(df['cluster'] == i) & (df['sexo'] == 'F')]) for i in range(4)]
        
        x = range(4)
        
        bars1 = ax.bar(x, masculino, label='Masculino', color='#4a9eff', alpha=0.8)
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
        ax.set_title('Distribuição de Sexo por Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(clusterNames)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        return self._figToBytes(fig)
    
    
    def _figToBytes(self, fig) -> bytes:
        """Converte figura matplotlib para bytes PNG"""
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    
    def generateZip(self, csvData: bytes, graphs: dict[str, bytes], 
                   pfd: bytes, includeReadme: bool = True) -> bytes:
        """
        Gera arquivo ZIP com CSV e gráficos.
        
        Args:
            csvData: Bytes do arquivo CSV
            graphs: Dicionário com gráficos {nome: bytes}
            includeReadme: Se deve incluir arquivo README
            
        Returns:
            Bytes do arquivo ZIP
        """
        
        zipBuffer = io.BytesIO()
        
        with zipfile.ZipFile(zipBuffer, 'w', zipfile.ZIP_DEFLATED) as zipFile:
            zipFile.writestr('dados/athletes.csv', csvData)
            zipFile.writestr('relatorio/report.pdf', pfd)
            
            for filename, imageBytes in graphs.items():
                zipFile.writestr(f'graficos/{filename}', imageBytes)
            
            if includeReadme:
                readme = self._generateReadme(len(graphs))
                zipFile.writestr('README.txt', readme)
        
        zipBuffer.seek(0)
        return zipBuffer.read()
    
    
    def _generateReadme(self, numGraphs: int) -> str:
        """Gera conteúdo do arquivo README"""
        
        return f"""
        ===========================================
        EXPORTAÇÃO DE DADOS - TALENT SCOUT
        ===========================================

        Data de Exportação: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}

        CONTEÚDO DO PACOTE:
        -------------------
        📁 dados/
        └── athletes.csv - Dados completos dos atletas

        📁 graficos/ ({numGraphs} arquivos)
        └── Gráficos estatísticos em alta resolução (300 DPI)

        DESCRIÇÃO DOS GRÁFICOS:
        -----------------------
        01_distribuicao_clusters.png    - Distribuição de atletas por cluster
        02_boxplot_features.png          - Box plots das features por cluster
        03_correlacao_features.png       - Matriz de correlação entre features
        04_altura_cluster.png            - Altura média por cluster e sexo
        05_arremesso_vs_salto.png        - Relação potência superior vs inferior
        06_sexo_cluster.png              - Distribuição de sexo por cluster

        CLUSTERS:
        ---------
        • Elite: Atletas de alto nível
        • Competitivo: Atletas experientes
        • Intermediário: Praticantes regulares
        • Iniciante: Novatos

        Para mais informações, visite: https://github.com/seu-projeto

        ===========================================
        """
    
    
    def exportData(self, athletesIds: list[str] = None, 
                  fullData: bool = True) -> tuple[Literal[True, False, -1], bytes | str]:
        """
        Exporta dados completos em formato ZIP.
        
        Args:
            athletesIds: Lista de IDs específicos (opcional)
            fullData: Se True, inclui gráficos. Se False, apenas CSV
            
        Returns:
            Tupla (status, dados_ou_mensagem)
            status: True (sucesso), False (erro leve), -1 (erro grave)
        """
        
        try:
            athletes = self.getAthletes(athletesIds)
            
            if len(athletes) == 0:
                return False, 'Não há dados para exportar'
            
            df = self.convertAthletesData(athletes)
            
            csvData = self.generateCSV(df)
            
            if not fullData:
                return True, csvData
            
            graphs = self.generateGraphs(df)
            
            zipData = self.generateZip(csvData, graphs, includeReadme=True)
            
            return True, zipData
            
        except Exception as e:
            errorMsg = (f'{type(e).__name__}: {e} '
                       f'in line {sys.exc_info()[-1].tb_lineno} '
                       f'in file {sys.exc_info()[-1].tb_frame.f_code.co_filename}')
            return -1, errorMsg