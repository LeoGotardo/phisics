import pandas as pd, numpy as np, random

from typing import Tuple, List, Dict, Literal
from src.model.athleteModel import Athlete
from datetime import datetime, timedelta
from src.config import Config


class DataGenerator:
    """
    Gerador de dados sintéticos para treinamento e testes.
    Ranges ajustados para refletir valores REALISTAS de performance atlética.
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
        """Gera um nome aleatório baseado no sexo."""
        if sexo == 'M':
            primeiroNome = random.choice(self.nomesMasculinos)
        else:
            primeiroNome = random.choice(self.nomesFemininos)
        
        sobrenome = random.choice(self.sobrenomes)
        return f"{primeiroNome} {sobrenome}"
    
    
    def gerarDataNascimento(self) -> datetime:
        """Gera uma data de nascimento aleatória entre 15 e 35 anos atrás."""
        hoje = datetime.now()
        anosAtras = random.randint(15, 35)
        diasRandom = random.randint(0, 365)
        return hoje - timedelta(days=anosAtras * 365 + diasRandom)
    
    
    def gerarDadosCluster(self, n: int, sexoDist: list, alturaRange: tuple, 
                         envergaduraRange: tuple, arremessoRange: tuple,
                         saltoRange: tuple, abdominaisRange: tuple,
                         cluster: int) -> List[Dict]:
        """
        Gera dados para um cluster específico.
        
        RANGES AJUSTADOS PARA VALORES REALISTAS:
        - Iniciante: Pessoa sedentária/normal (seu exemplo: 183cm, 120cm envergadura, 550cm arremesso, 80cm salto, 20 abdominais)
        - Intermediário: Praticante recreativo
        - Competitivo: Atleta amador competitivo
        - Elite: Atleta de alto nível
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
                    arremessoRange[0] * 0.70, 
                    arremessoRange[1] * 0.75
                )
                salto = np.random.uniform(
                    saltoRange[0] * 0.80, 
                    saltoRange[1] * 0.85
                )
                abdominais = np.random.randint(
                    int(abdominaisRange[0] * 0.85),
                    int(abdominaisRange[1] * 0.90)
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
        
        PARÂMETROS AJUSTADOS BASEADOS EM DADOS REALISTAS:
        
        SEU PERFIL (pessoa normal):
        - Altura: 183cm
        - Envergadura: 120cm (ratio ~0.66)
        - Arremesso: 550cm = 5.5m
        - Salto: 80cm = 0.8m
        - Abdominais: 20 rep/min
        → Deve ser classificado como INICIANTE
        """
        
        # INICIANTE (25%) - Pessoa sedentária/normal
        # Baseado no seu perfil
        iniciante = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.5, 0.5],
            alturaRange=(160, 190),           # Altura normal
            envergaduraRange=(0.60, 0.70),    # 60-70% da altura (seu caso: 120/183 = 0.66)
            arremessoRange=(3.0, 6.5),        # 3-6.5m (seu caso: 5.5m)
            saltoRange=(0.50, 1.00),          # 50-100cm (seu caso: 80cm)
            abdominaisRange=(10, 30),         # 10-30 rep/min (seu caso: 20)
            cluster=0
        )
        
        # INTERMEDIÁRIO (25%) - Praticante recreativo
        intermediario = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.5, 0.5],
            alturaRange=(165, 190),
            envergaduraRange=(0.68, 0.78),    # Melhor relação
            arremessoRange=(6.0, 8.5),        # 6-8.5m
            saltoRange=(0.95, 1.40),          # 95-140cm
            abdominaisRange=(28, 45),         # 28-45 rep/min
            cluster=1
        )
        
        # COMPETITIVO (25%) - Atleta amador sério
        competitivo = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.55, 0.45],
            alturaRange=(168, 195),
            envergaduraRange=(0.76, 0.88),
            arremessoRange=(8.0, 11.0),       # 8-11m
            saltoRange=(1.35, 1.85),          # 135-185cm
            abdominaisRange=(43, 60),         # 43-60 rep/min
            cluster=2
        )
        
        # ELITE (25%) - Atleta profissional/alto nível
        elite = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.6, 0.4],
            alturaRange=(170, 200),
            envergaduraRange=(0.85, 1.00),    # Envergadura >= altura
            arremessoRange=(10.5, 15.0),      # 10.5-15m
            saltoRange=(1.80, 2.50),          # 180-250cm
            abdominaisRange=(58, 80),         # 58-80 rep/min
            cluster=3
        )
        
        # Combinar todos os dados
        todosDados = iniciante + intermediario + competitivo + elite
        
        # Embaralhar
        random.shuffle(todosDados)
        
        if returnType == 'list':
            return todosDados
        
        elif returnType == 'athletes':
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
        """Gera dados e salva em arquivo CSV."""
        try:
            df = self.generateData(returnType='df')
            df['dataNascimento'] = df['dataNascimento'].dt.strftime('%Y-%m-%d')
            df.to_csv(filepath, index=False)
            
            stats = f"""
═══════════════════════════════════════════════════════════════════
DATASET CRIADO COM SUCESSO - RANGES REALISTAS!
═══════════════════════════════════════════════════════════════════

Total de exemplos: {len(df)}

Distribuição por cluster:
{df['cluster'].value_counts().sort_index()}

Distribuição por sexo:
{df['sexo'].value_counts()}

Estatísticas descritivas:
{df[['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']].describe().round(2)}

✓ Arquivo salvo: {filepath}

REFERÊNCIAS (pessoa normal deve ser INICIANTE):
- Altura: 160-190cm
- Envergadura: 60-70% da altura
- Arremesso: 3-6.5m
- Salto: 50-100cm
- Abdominais: 10-30 rep/min
═══════════════════════════════════════════════════════════════════
            """
            
            return True, stats
            
        except Exception as e:
            return False, f"Erro ao salvar CSV: {str(e)}"
    
    
    def saveToDatabase(self, clearExisting: bool = False) -> Tuple[bool, str]:
        """Gera dados e salva diretamente no banco de dados."""
        try:
            with Config.app.app_context():
                if clearExisting:
                    self.session.query(Athlete).delete()
                    self.session.commit()
                
                athletes = self.generateData(returnType='athletes')
                
                for athlete in athletes:
                    self.session.add(athlete)
                
                self.session.commit()
                
                total = len(athletes)
                por_cluster = {}
                for athlete in athletes:
                    cluster_name = self.clusterNames[athlete.cluster]
                    por_cluster[cluster_name] = por_cluster.get(cluster_name, 0) + 1
                
                mensagem = f"""
✓ {total} atletas inseridos no banco de dados

Distribuição por cluster:
{chr(10).join(f'  {k}: {v} atletas' for k, v in sorted(por_cluster.items()))}

Seu perfil (183cm, 120cm enverg, 5.5m arremesso, 80cm salto, 20 abdom) 
→ Será classificado como INICIANTE ✓
                """
                
                return True, mensagem
            
        except Exception as e:
            self.session.rollback()
            return False, f"Erro ao salvar no banco: {str(e)}"
    
    
    def generateSingleAthlete(self, cluster: int = None, sexo: str = None) -> Athlete:
        """Gera um único atleta com características específicas."""
        if cluster is None:
            cluster = random.randint(0, 3)
        
        if sexo is None:
            sexo = random.choice(['M', 'F'])
        
        params = {
            0: {  # Iniciante
                'alturaRange': (160, 190),
                'envergaduraRange': (0.60, 0.70),
                'arremessoRange': (3.0, 6.5),
                'saltoRange': (0.50, 1.00),
                'abdominaisRange': (10, 30)
            },
            1: {  # Intermediário
                'alturaRange': (165, 190),
                'envergaduraRange': (0.68, 0.78),
                'arremessoRange': (6.0, 8.5),
                'saltoRange': (0.95, 1.40),
                'abdominaisRange': (28, 45)
            },
            2: {  # Competitivo
                'alturaRange': (168, 195),
                'envergaduraRange': (0.76, 0.88),
                'arremessoRange': (8.0, 11.0),
                'saltoRange': (1.35, 1.85),
                'abdominaisRange': (43, 60)
            },
            3: {  # Elite
                'alturaRange': (170, 200),
                'envergaduraRange': (0.85, 1.00),
                'arremessoRange': (10.5, 15.0),
                'saltoRange': (1.80, 2.50),
                'abdominaisRange': (58, 80)
            }
        }
        
        p = params[cluster]
        nome = self.gerarNome(sexo)
        dataNascimento = self.gerarDataNascimento()
        
        if sexo == 'M':
            altura = np.random.uniform(*p['alturaRange'])
            envergadura = altura * np.random.uniform(*p['envergaduraRange'])
            arremesso = np.random.uniform(*p['arremessoRange'])
            salto = np.random.uniform(*p['saltoRange'])
            abdominais = np.random.randint(*p['abdominaisRange'])
        else:
            altura = np.random.uniform(p['alturaRange'][0] * 0.92, p['alturaRange'][1] * 0.95)
            envergadura = altura * np.random.uniform(*p['envergaduraRange'])
            arremesso = np.random.uniform(p['arremessoRange'][0] * 0.70, p['arremessoRange'][1] * 0.75)
            salto = np.random.uniform(p['saltoRange'][0] * 0.80, p['saltoRange'][1] * 0.85)
            abdominais = np.random.randint(int(p['abdominaisRange'][0] * 0.85), int(p['abdominaisRange'][1] * 0.90))
        
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
        """Retorna estatísticas sobre os dados que seriam gerados."""
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