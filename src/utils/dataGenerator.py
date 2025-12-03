import pandas as pd, numpy as np, random

from typing import Tuple, List, Dict, Literal
from src.model.athleteModel import Athlete
from datetime import datetime, timedelta
from src.config import Config


class DataGenerator:
    """
    Gerador de dados sintéticos para treinamento e testes.
    Gera dados realistas com distribuições adequadas para cada cluster.
    Compatível com o modelo Athlete e banco de dados do projeto.
    """
    
    def __init__(self, nAthletes: int = 160):
        """
        Inicializa o gerador de dados.
        
        Args:
            nAthletes: Número total de atletas a gerar
        """
        self.nAthletes = nAthletes
        self.session = Config.session
        
        self.nomesMasculinos = [
            'João', 'Pedro', 'Lucas', 'Gabriel', 'Matheus', 'Rafael', 'Bruno', 
            'Felipe', 'Guilherme', 'Rodrigo', 'Diego', 'Fernando', 'André', 
            'Carlos', 'Eduardo', 'Thiago', 'Marcelo', 'Daniel', 'Leonardo', 'Paulo'
        ]
        
        self.nomesFemininos = [
            'Maria', 'Ana', 'Juliana', 'Fernanda', 'Carla', 'Beatriz', 'Camila',
            'Amanda', 'Paula', 'Larissa', 'Mariana', 'Patrícia', 'Renata', 
            'Gabriela', 'Carolina', 'Vanessa', 'Bianca', 'Letícia', 'Natália', 'Débora'
        ]
        
        self.sobrenomes = [
            'Silva', 'Santos', 'Oliveira', 'Souza', 'Costa', 'Ferreira', 'Rodrigues',
            'Alves', 'Pereira', 'Lima', 'Gomes', 'Martins', 'Ribeiro', 'Carvalho',
            'Rocha', 'Almeida', 'Nascimento', 'Araújo', 'Fernandes', 'Barbosa'
        ]
        
        self.clusterNames = {
            0: 'Iniciante',
            1: 'Intermediário',
            2: 'Competitivo',
            3: 'Elite'
        }
    
    
    def gerarNome(self, sexo: str) -> str:
        """
        Gera um nome aleatório baseado no sexo.
        
        Args:
            sexo: 'M' ou 'F'
            
        Returns:
            Nome completo gerado
        """
        if sexo == 'M':
            primeiroNome = random.choice(self.nomesMasculinos)
        else:
            primeiroNome = random.choice(self.nomesFemininos)
        
        sobrenome = random.choice(self.sobrenomes)
        return f"{primeiroNome} {sobrenome}"
    
    
    def gerarDataNascimento(self) -> datetime:
        """
        Gera uma data de nascimento aleatória entre 15 e 35 anos atrás.
        
        Returns:
            Data de nascimento gerada
        """
        hoje = datetime.now()
        anosAtras = random.randint(15, 35)
        diasRandom = random.randint(0, 365)
        return hoje - timedelta(days=anosAtras * 365 + diasRandom)
    
    
    def gerarDadosCluster(self, n: int, sexoDist: list, alturaRange: tuple, envergaduraRange: tuple, arremessoRange: tuple,
                         saltoRange: tuple, abdominaisRange: tuple,
                         cluster: int) -> List[Dict]:
        """
        Gera dados para um cluster específico.
        
        Args:
            n: Número de atletas
            sexoDist: [prob_masculino, prob_feminino]
            alturaRange: (min, max) altura em cm
            envergaduraRange: (min, max) envergadura em cm
            arremessoRange: (min, max) arremesso em metros
            saltoRange: (min, max) salto horizontal em metros
            abdominaisRange: (min, max) abdominais em repetições
            cluster: ID do cluster (0-3)
            
        Returns:
            Lista de dicionários com dados dos atletas
        """
        dados = []
        
        for _ in range(n):
            sexo = np.random.choice(['M', 'F'], p=sexoDist)
            nome = self.gerarNome(sexo)
            dataNascimento = self.gerarDataNascimento()
            
            if sexo == 'M':
                altura = np.random.uniform(*alturaRange)
                envergadura = altura * np.random.uniform(*envergaduraRange)
                arremesso = np.random.uniform(*arremessoRange)
                salto = np.random.uniform(*saltoRange)
                abdominais = np.random.randint(*abdominaisRange)
            else:
                # Mulheres geralmente têm valores proporcionalmente menores
                altura = np.random.uniform(
                    alturaRange[0] * 0.92, 
                    alturaRange[1] * 0.95
                )
                envergadura = altura * np.random.uniform(*envergaduraRange)
                arremesso = np.random.uniform(
                    arremessoRange[0] * 0.7, 
                    arremessoRange[1] * 0.75
                )
                salto = np.random.uniform(
                    saltoRange[0] * 0.8, 
                    saltoRange[1] * 0.85
                )
                abdominais = np.random.randint(
                    int(abdominaisRange[0] * 0.85),
                    int(abdominaisRange[1] * 0.9)
                )
            
            dados.append({
                'nome': nome,
                'dataNascimento': dataNascimento,
                'sexo': sexo,
                'altura': round(altura, 2),
                'envergadura': round(envergadura, 2),
                'arremesso': round(arremesso, 2),
                'saltoHorizontal': round(salto, 2),
                'abdominais': int(abdominais),
                'cluster': cluster
            })
        
        return dados
    
    
    def generateData(self, returnType: Literal['df', 'list', 'athletes'] = 'df') -> pd.DataFrame | List[Dict] | List[Athlete]:
        """
        Gera o dataset completo com todos os clusters.
        
        Args:
            returnType: 
                - 'df': retorna pandas DataFrame
                - 'list': retorna lista de dicionários
                - 'athletes': retorna lista de objetos Athlete
            
        Returns:
            Dados gerados no formato especificado
        """
        # Elite (25%)
        elite = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.6, 0.4],
            alturaRange=(165, 185),
            envergaduraRange=(170, 195),
            arremessoRange=(10, 14),
            saltoRange=(2.6, 3.2),
            abdominaisRange=(55, 75),
            cluster=3
        )
        
        # Competitivo (25%)
        competitivo = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.55, 0.45],
            alturaRange=(160, 182),
            envergaduraRange=(165, 190),
            arremessoRange=(7.5, 10.5),
            saltoRange=(2.1, 2.7),
            abdominaisRange=(40, 58),
            cluster=2
        )
        
        # Intermediário (25%)
        intermediario = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.5, 0.5],
            alturaRange=(158, 180),
            envergaduraRange=(160, 185),
            arremessoRange=(5.5, 8.0),
            saltoRange=(1.7, 2.3),
            abdominaisRange=(28, 45),
            cluster=1
        )
        
        # Iniciante (25%)
        iniciante = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.5, 0.5],
            alturaRange=(155, 178),
            envergaduraRange=(158, 182),
            arremessoRange=(3.5, 6.0),
            saltoRange=(1.2, 1.8),
            abdominaisRange=(15, 32),
            cluster=0
        )
        
        # Combinar todos os dados
        todosDados = elite + competitivo + intermediario + iniciante
        
        # Embaralhar
        random.shuffle(todosDados)
        
        if returnType == 'list':
            return todosDados
        
        elif returnType == 'athletes':
            # Converter para objetos Athlete
            athletes = []
            for data in todosDados:
                athlete = Athlete(
                    nome=data['nome'],
                    dataNascimento=data['dataNascimento'],
                    sexo=data['sexo'],
                    altura=data['altura'],
                    envergadura=data['envergadura'],
                    arremesso=data['arremesso'],
                    saltoHorizontal=data['saltoHorizontal'],
                    abdominais=data['abdominais']
                )
                athlete.cluster = data['cluster']
                athletes.append(athlete)
            
            return athletes
        
        else:  # returnType == 'df'
            df = pd.DataFrame(todosDados)
            return df
    
    
    def saveToCSV(self, filepath: str = 'dataset_athletes.csv') -> Tuple[bool, str]:
        """
        Gera dados e salva em arquivo CSV.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            df = self.generateData(returnType='df')
            
            # Converter dataNascimento para string
            df['dataNascimento'] = df['dataNascimento'].dt.strftime('%Y-%m-%d')
            
            df.to_csv(filepath, index=False)
            
            # Estatísticas
            stats = f"""
═══════════════════════════════════════════════════════════════════
DATASET CRIADO COM SUCESSO!
═══════════════════════════════════════════════════════════════════

Total de exemplos: {len(df)}

Distribuição por cluster:
{df['cluster'].value_counts().sort_index()}

Distribuição por sexo:
{df['sexo'].value_counts()}

Estatísticas descritivas:
{df[['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']].describe().round(2)}

✓ Arquivo salvo: {filepath}
═══════════════════════════════════════════════════════════════════
            """
            
            return True, stats
            
        except Exception as e:
            return False, f"Erro ao salvar CSV: {str(e)}"
    
    
    def saveToDatabase(self, clearExisting: bool = False) -> Tuple[bool, str]:
        """
        Gera dados e salva diretamente no banco de dados.
        
        Args:
            clearExisting: Se True, limpa dados existentes antes de inserir
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            with Config.app.app_context():
                # Limpar dados existentes se solicitado
                if clearExisting:
                    self.session.query(Athlete).delete()
                    self.session.commit()
                
                # Gerar atletas
                athletes = self.generateData(returnType='athletes')
                
                # Inserir no banco
                for athlete in athletes:
                    self.session.add(athlete)
                
                self.session.commit()
                
                # Estatísticas
                total = len(athletes)
                por_cluster = {}
                for athlete in athletes:
                    cluster_name = self.clusterNames[athlete.cluster]
                    por_cluster[cluster_name] = por_cluster.get(cluster_name, 0) + 1
                
                mensagem = f"""
✓ {total} atletas inseridos no banco de dados

Distribuição por cluster:
{chr(10).join(f'  {k}: {v} atletas' for k, v in sorted(por_cluster.items()))}
                """
                
                return True, mensagem
            
        except Exception as e:
            self.session.rollback()
            return False, f"Erro ao salvar no banco: {str(e)}"
    
    
    def generateSingleAthlete(self, cluster: int = None, sexo: str = None) -> Athlete:
        """
        Gera um único atleta com características específicas.
        
        Args:
            cluster: Cluster desejado (0-3). Se None, escolhe aleatoriamente
            sexo: 'M' ou 'F'. Se None, escolhe aleatoriamente
            
        Returns:
            Objeto Athlete gerado
        """
        if cluster is None:
            cluster = random.randint(0, 3)
        
        if sexo is None:
            sexo = random.choice(['M', 'F'])
        
        # Parâmetros por cluster
        params = {
            3: {  # Elite
                'alturaRange': (165, 185),
                'envergaduraRange': (170, 195),
                'arremessoRange': (10, 14),
                'saltoRange': (2.6, 3.2),
                'abdominaisRange': (55, 75)
            },
            2: {  # Competitivo
                'alturaRange': (160, 182),
                'envergaduraRange': (165, 190),
                'arremessoRange': (7.5, 10.5),
                'saltoRange': (2.1, 2.7),
                'abdominaisRange': (40, 58)
            },
            1: {  # Intermediário
                'alturaRange': (158, 180),
                'envergaduraRange': (160, 185),
                'arremessoRange': (5.5, 8.0),
                'saltoRange': (1.7, 2.3),
                'abdominaisRange': (28, 45)
            },
            0: {  # Iniciante
                'alturaRange': (155, 178),
                'envergaduraRange': (158, 182),
                'arremessoRange': (3.5, 6.0),
                'saltoRange': (1.2, 1.8),
                'abdominaisRange': (15, 32)
            }
        }
        
        p = params[cluster]
        
        # Gerar dados
        nome = self.gerarNome(sexo)
        dataNascimento = self.gerarDataNascimento()
        
        if sexo == 'M':
            altura = np.random.uniform(*p['alturaRange'])
            envergadura = altura * np.random.uniform(1.0, 1.06)
            arremesso = np.random.uniform(*p['arremessoRange'])
            salto = np.random.uniform(*p['saltoRange'])
            abdominais = np.random.randint(*p['abdominaisRange'])
        else:
            altura = np.random.uniform(
                p['alturaRange'][0] * 0.92,
                p['alturaRange'][1] * 0.95
            )
            envergadura = altura * np.random.uniform(1.0, 1.05)
            arremesso = np.random.uniform(
                p['arremessoRange'][0] * 0.7,
                p['arremessoRange'][1] * 0.75
            )
            salto = np.random.uniform(
                p['saltoRange'][0] * 0.8,
                p['saltoRange'][1] * 0.85
            )
            abdominais = np.random.randint(
                int(p['abdominaisRange'][0] * 0.85),
                int(p['abdominaisRange'][1] * 0.9)
            )
        
        # Criar atleta
        athlete = Athlete(
            nome=nome,
            dataNascimento=dataNascimento,
            sexo=sexo,
            altura=round(altura, 2),
            envergadura=round(envergadura, 2),
            arremesso=round(arremesso, 2),
            saltoHorizontal=round(salto, 2),
            abdominais=int(abdominais)
        )
        
        athlete.cluster = cluster
        
        return athlete
    
    
    def getStatistics(self) -> Dict:
        """
        Retorna estatísticas sobre os dados que seriam gerados.
        
        Returns:
            Dicionário com estatísticas
        """
        df = self.generateData(returnType='df')
        
        stats = {
            'total': len(df),
            'por_cluster': df['cluster'].value_counts().to_dict(),
            'por_sexo': df['sexo'].value_counts().to_dict(),
            'medias_por_cluster': df.groupby('cluster')[
                ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']
            ].mean().round(2).to_dict(),
            'desvios_por_cluster': df.groupby('cluster')[
                ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']
            ].std().round(2).to_dict()
        }
        
        return stats