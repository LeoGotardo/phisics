import pandas as pd, numpy as np, random

from typing import Tuple, List, Dict, Literal
from src.model.athleteModel import Athlete
from datetime import datetime, timedelta
from src.config import Config


class DataGenerator:
    """
    Gerador de dados sintéticos REALISTAS para treinamento e testes.
    Baseado em dados reais de testes físicos e antropometria.
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
            'Carlos', 'Eduardo', 'Thiago', 'Marcelo', 'Daniel', 'Leonardo', 'Paulo',
            'Vitor', 'Caio', 'Henrique', 'Ricardo', 'Alexandre', 'Fábio', 'Gustavo',
            'Vinícius', 'Leandro', 'Maurício'
        ]
        
        self.nomesFemininos = [
            'Maria', 'Ana', 'Juliana', 'Fernanda', 'Carla', 'Beatriz', 'Camila',
            'Amanda', 'Paula', 'Larissa', 'Mariana', 'Patrícia', 'Renata', 
            'Gabriela', 'Carolina', 'Vanessa', 'Bianca', 'Letícia', 'Natália', 'Débora',
            'Tatiana', 'Priscila', 'Rafaela', 'Bruna', 'Cristina', 'Aline', 'Jéssica',
            'Isabela', 'Luciana', 'Daniela'
        ]
        
        self.sobrenomes = [
            'Silva', 'Santos', 'Oliveira', 'Souza', 'Costa', 'Ferreira', 'Rodrigues',
            'Alves', 'Pereira', 'Lima', 'Gomes', 'Martins', 'Ribeiro', 'Carvalho',
            'Rocha', 'Almeida', 'Nascimento', 'Araújo', 'Fernandes', 'Barbosa',
            'Mendes', 'Cardoso', 'Reis', 'Campos', 'Moreira', 'Teixeira', 'Correia',
            'Castro', 'Pinto', 'Soares'
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
        segundoSobrenome = random.choice(self.sobrenomes)
        
        # 30% de chance de ter dois sobrenomes
        if random.random() < 0.3:
            return f"{primeiroNome} {sobrenome} {segundoSobrenome}"
        
        return f"{primeiroNome} {sobrenome}"
    
    
    def gerarDataNascimento(self, idadeRange: tuple = (15, 35)) -> datetime:
        """
        Gera uma data de nascimento aleatória.
        
        Args:
            idadeRange: Tupla (idade_min, idade_max)
        """
        hoje = datetime.now()
        idade = random.randint(idadeRange[0], idadeRange[1])
        
        # Distribuição mais realista: mais jovens
        if random.random() < 0.4:  # 40% entre 18-25
            idade = random.randint(18, 25)
        elif random.random() < 0.7:  # 30% entre 26-30
            idade = random.randint(26, 30)
        
        diasRandom = random.randint(0, 365)
        mesRandom = random.randint(1, 12)
        diaRandom = random.randint(1, 28)
        
        dataNascimento = hoje.replace(year=hoje.year - idade, month=mesRandom, day=diaRandom)
        
        return dataNascimento
    
    
    def gerarAlturaEnvergadura(self, sexo: str, cluster: int) -> tuple:
        """
        Gera altura e envergadura correlacionadas de forma realista.
        
        Referências antropométricas:
        - Envergadura geralmente = altura ± 5cm
        - Homens: média 175cm (DP: 7cm)
        - Mulheres: média 162cm (DP: 6cm)
        """
        
        if sexo == 'M':
            # Altura base por cluster (cm)
            alturaBase = {
                0: 175,  # Iniciante: população geral
                1: 180,  # Intermediário: ligeiramente acima
                2: 191,  # Competitivo: mais altos
                3: 214   # Elite: maiores alturas
            }
            
            desvioPadrao = 7
            
        else:  # Feminino
            alturaBase = {
                0: 162,
                1: 165,
                2: 176,
                3: 185
            }
            
            desvioPadrao = 6
        
        # Gerar altura com distribuição normal
        altura = np.random.normal(alturaBase[cluster], desvioPadrao)
        altura = max(150, min(210, altura))  # Limites realistas
        
        # Envergadura: geralmente 102-105% da altura
        ratioEnvergadura = np.random.normal(1.03, 0.03)  # Média 103%, DP 3%
        ratioEnvergadura = max(0.97, min(1.08, ratioEnvergadura))
        
        envergadura = altura * ratioEnvergadura
        
        return round(altura, 1), round(envergadura, 1)
    
    
    def gerarArremessoBall(self, sexo: str, cluster: int, altura: float) -> float:
        """
        Gera resultado de arremesso de medicine ball (em metros).
        
        Referências:
        - Teste padrão: medicine ball 3kg para homens, 2kg para mulheres
        - Homens adultos: 4-12m (média ~6.5m)
        - Mulheres adultas: 3-9m (média ~5m)
        """
        
        if sexo == 'M':
            # Médias por cluster (metros)
            mediaBase = {
                0: 5.5,   # Iniciante: sedentário
                1: 7.5,   # Intermediário: praticante
                2: 9.5,   # Competitivo: atleta amador
                3: 11.5   # Elite: atleta profissional
            }
            desvio = 1.2
            
        else:
            mediaBase = {
                0: 4.0,
                1: 5.5,
                2: 7.0,
                3: 8.5
            }
            desvio = 0.9
        
        # Influência da altura (+5% por cada 10cm acima de 170cm para homens / 160cm para mulheres)
        alturaRef = 170 if sexo == 'M' else 160
        bonusAltura = (altura - alturaRef) / 10 * 0.05
        
        arremesso = np.random.normal(mediaBase[cluster] * (1 + bonusAltura), desvio)
        arremesso = max(2.5, min(15.0, arremesso))
        
        return round(arremesso, 2)
    
    
    def gerarSaltoHorizontal(self, sexo: str, cluster: int, altura: float) -> float:
        """
        Gera resultado de salto horizontal (em metros).
        
        Referências:
        - Homens adultos: 1.5-2.8m (média ~2.0m)
        - Mulheres adultas: 1.2-2.2m (média ~1.6m)
        - Atletas elite: >2.5m (M), >2.0m (F)
        """
        
        if sexo == 'M':
            mediaBase = {
                0: 1.65,  # Iniciante: pessoa comum
                1: 2.00,  # Intermediário: praticante
                2: 2.35,  # Competitivo: atleta
                3: 2.65   # Elite: alto nível
            }
            desvio = 0.18
            
        else:
            mediaBase = {
                0: 1.35,
                1: 1.65,
                2: 1.90,
                3: 2.15
            }
            desvio = 0.15
        
        # Influência da altura
        alturaRef = 175 if sexo == 'M' else 165
        bonusAltura = (altura - alturaRef) / 10 * 0.03
        
        salto = np.random.normal(mediaBase[cluster] * (1 + bonusAltura), desvio)
        salto = max(0.8, min(3.2, salto))
        
        return round(salto, 2)
    
    
    def gerarAbdominais(self, sexo: str, cluster: int, idade: int) -> int:
        """
        Gera resultado de abdominais em 1 minuto.
        
        Referências (teste sit-ups padrão):
        - Homens 20-29 anos: 15-50 rep/min (média ~35)
        - Mulheres 20-29 anos: 10-45 rep/min (média ~30)
        - Declínio de ~5% por década após 30 anos
        """
        
        if sexo == 'M':
            mediaBase = {
                0: 25,   # Iniciante: sedentário
                1: 38,   # Intermediário: praticante
                2: 50,   # Competitivo: atleta
                3: 62    # Elite: alto nível
            }
            desvio = 5
            
        else:
            mediaBase = {
                0: 20,
                1: 33,
                2: 44,
                3: 55
            }
            desvio = 4
        
        # Ajuste por idade
        if idade > 30:
            penalidade = (idade - 30) / 10 * 0.05
            mediaBase[cluster] = int(mediaBase[cluster] * (1 - penalidade))
        
        abdominais = np.random.normal(mediaBase[cluster], desvio)
        abdominais = max(8, min(80, abdominais))
        
        return int(round(abdominais))
    
    
    def gerarDadosCluster(self, n: int, sexoDist: list, cluster: int) -> List[Dict]:
        """
        Gera dados para um cluster específico com correlações realistas.
        
        Args:
            n: Número de atletas
            sexoDist: Distribuição de sexo [prob_masculino, prob_feminino]
            cluster: ID do cluster (0-3)
        """
        dados = []
        
        for _ in range(n):
            sexo = np.random.choice(['M', 'F'], p=sexoDist)
            nome = self.gerarNome(sexo)
            dataNascimento = self.gerarDataNascimento()
            idade = (datetime.now() - dataNascimento).days // 365
            
            # Gerar dados correlacionados
            altura, envergadura = self.gerarAlturaEnvergadura(sexo, cluster)
            arremesso = self.gerarArremessoBall(sexo, cluster, altura)
            salto = self.gerarSaltoHorizontal(sexo, cluster, altura)
            abdominais = self.gerarAbdominais(sexo, cluster, idade)
            
            dados.append({
                'nome': nome,
                'dataNascimento': dataNascimento,
                'sexo': sexo,
                'altura': altura,
                'envergadura': envergadura,
                'arremesso': arremesso,
                'saltoHorizontal': salto,
                'abdominais': abdominais,
                'cluster': cluster
            })
        
        return dados
    
    
    def generateData(self, returnType: Literal['df', 'list', 'athletes'] = 'df') -> pd.DataFrame | List[Dict] | List[Athlete]:
        """
        Gera o dataset completo com todos os clusters e correlações realistas.
        
        Distribuição:
        - Iniciante: 30% (maioria da população)
        - Intermediário: 35% (praticantes regulares)
        - Competitivo: 25% (atletas amadores)
        - Elite: 10% (alto nível)
        """
        
        # INICIANTE (30%) - População geral / sedentários
        iniciante = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.30),
            sexoDist=[0.5, 0.5],
            cluster=0
        )
        
        # INTERMEDIÁRIO (35%) - Praticantes regulares
        intermediario = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.35),
            sexoDist=[0.52, 0.48],
            cluster=1
        )
        
        # COMPETITIVO (25%) - Atletas amadores competitivos
        competitivo = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.25),
            sexoDist=[0.58, 0.42],
            cluster=2
        )
        
        # ELITE (10%) - Atletas de alto nível
        elite = self.gerarDadosCluster(
            n=int(self.nAthletes * 0.10),
            sexoDist=[0.65, 0.35],
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
DATASET REALISTA CRIADO COM SUCESSO!
═══════════════════════════════════════════════════════════════════

Total de atletas: {len(df)}

Distribuição por cluster:
{df['cluster'].value_counts().sort_index()}

Distribuição por sexo:
{df['sexo'].value_counts()}

Estatísticas por cluster:
{df.groupby('cluster')[['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']].mean().round(2)}

✓ Arquivo salvo: {filepath}

REFERÊNCIAS REALISTAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INICIANTE (sedentário/pessoa comum):
  Homem:   175cm / enverg 180cm / 5.5m / salto 1.65m / 25 abd
  Mulher:  162cm / enverg 167cm / 4.0m / salto 1.35m / 20 abd

INTERMEDIÁRIO (praticante regular):
  Homem:   178cm / enverg 183cm / 7.5m / salto 2.00m / 38 abd
  Mulher:  165cm / enverg 170cm / 5.5m / salto 1.65m / 33 abd

COMPETITIVO (atleta amador):
  Homem:   181cm / enverg 186cm / 9.5m / salto 2.35m / 50 abd
  Mulher:  167cm / enverg 172cm / 7.0m / salto 1.90m / 44 abd

ELITE (alto nível):
  Homem:   184cm / enverg 190cm / 11.5m / salto 2.65m / 62 abd
  Mulher:  170cm / enverg 175cm / 8.5m / salto 2.15m / 55 abd
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """
            
            return True, stats
            
        except Exception as e:
            return False, f"Erro ao salvar CSV: {str(e)}"
    
    
    def saveToDatabase(self, clearExisting: bool = False) -> Tuple[bool, str]:
        """Gera dados e salva diretamente no banco de dados."""
        with Config.app.app_context():
            try:
                if clearExisting:
                    self.session.query(Athlete).delete()
                    self.session.commit()
                
                athletes = self.generateData(returnType='athletes')
                
                for athlete in athletes:
                    self.session.add(athlete)
                
                self.session.commit()
                
                total = len(athletes)
                porCluster = {}
                for athlete in athletes:
                    clusterName = self.clusterNames[athlete.cluster]
                    porCluster[clusterName] = porCluster.get(clusterName, 0) + 1
                
                mensagem = f"""
✓ {total} atletas inseridos no banco de dados com DADOS REALISTAS

Distribuição por cluster:
{chr(10).join(f'  {k}: {v} atletas ({v/total*100:.1f}%)' for k, v in sorted(porCluster.items()))}

Dados baseados em:
  • Antropometria populacional real
  • Testes físicos padronizados (ACSM)
  • Correlações fisiológicas naturais
  • Distribuição etária realista (18-35 anos)
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
        
        nome = self.gerarNome(sexo)
        dataNascimento = self.gerarDataNascimento()
        idade = (datetime.now() - dataNascimento).days // 365
        
        altura, envergadura = self.gerarAlturaEnvergadura(sexo, cluster)
        arremesso = self.gerarArremessoBall(sexo, cluster, altura)
        salto = self.gerarSaltoHorizontal(sexo, cluster, altura)
        abdominais = self.gerarAbdominais(sexo, cluster, idade)
        
        athlete = Athlete(
            nome=nome,
            dataNascimento=dataNascimento,
            sexo=sexo,
            altura=altura,
            envergadura=envergadura,
            arremesso=arremesso,
            saltoHorizontal=salto,
            abdominais=abdominais
        )
        
        athlete.cluster = cluster
        return athlete
    
    
    def getStatistics(self) -> Dict:
        """Retorna estatísticas sobre os dados que seriam gerados."""
        df = self.generateData(returnType='df')
        
        stats = {
            'total': len(df),
            'porCluster': df['cluster'].value_counts().to_dict(),
            'porSexo': df['sexo'].value_counts().to_dict(),
            'mediasPorCluster': df.groupby('cluster')[
                ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']
            ].mean().round(2).to_dict(),
            'desviosPorCluster': df.groupby('cluster')[
                ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']
            ].std().round(2).to_dict()
        }
        
        return stats