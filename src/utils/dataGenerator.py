import pandas as pd, random as r, numpy as np

from src.model.athleteModel import Athlete
from src.utils.dataclasses import Column
from src.config import Config

class DataGenerator:
    def __init__(self, nAthletes: int = Config.N_ATHLETES) -> None:
        self.nAthletes = nAthletes
        self.athletes = pd.DataFrame()
        self.columns: list[Column] = Config.COLUMNS
                
        
    def gerar_dados(self, n, sexo_dist, altura_range, envergadura_range, 
                        arremesso_range, salto_range, abdominais_range, cluster):
            dados = []
            
            for _ in range(n):
                sexo = np.random.choice(['M', 'F'], p=sexo_dist)
                
                if sexo == 'M':
                    altura = np.random.uniform(*altura_range)
                    envergadura = altura * np.random.uniform(1.0, 1.06)
                    arremesso = np.random.uniform(*arremesso_range)
                    salto = np.random.uniform(*salto_range)
                    abdominais = np.random.randint(*abdominais_range)
                else:
                    altura = np.random.uniform(altura_range[0] * 0.92, altura_range[1] * 0.95)
                    envergadura = altura * np.random.uniform(1.0, 1.05)
                    arremesso = np.random.uniform(arremesso_range[0] * 0.7, arremesso_range[1] * 0.75)
                    salto = np.random.uniform(salto_range[0] * 0.8, salto_range[1] * 0.85)
                    abdominais = np.random.randint(int(abdominais_range[0] * 0.85), int(abdominais_range[1] * 0.9))
                
                dados.append({
                    'sexo': sexo,
                    'altura': round(altura, 2),
                    'envergadura': round(envergadura, 2),
                    'arremesso': round(arremesso, 2),
                    'saltoHorizontal': round(salto, 2),
                    'abdominais': abdominais,
                    'cluster': cluster
                })
            
            return dados
    
    
    def generateData(self, returnType: str = 'df') -> pd.DataFrame | list:
        # Função para gerar dados com distribuição realista

        elite = self.gerar_dados(
            n=40,
            sexo_dist=[0.6, 0.4],  # 60% M, 40% F
            altura_range=(1.65, 1.85),
            envergadura_range=(1.70, 1.95),
            arremesso_range=(10, 14),
            salto_range=(2.6, 3.2),
            abdominais_range=(55, 75),
            cluster='Elite'
        )

        competitivo = self.gerar_dados(
            n=40,
            sexo_dist=[0.55, 0.45],
            altura_range=(1.60, 1.82),
            envergadura_range=(1.65, 1.90),
            arremesso_range=(7.5, 10.5),
            salto_range=(2.1, 2.7),
            abdominais_range=(40, 58),
            cluster='Competitivo'
        )

        intermediario = self.gerar_dados(
            n=40,
            sexo_dist=[0.5, 0.5],
            altura_range=(1.58, 1.80),
            envergadura_range=(1.60, 1.85),
            arremesso_range=(5.5, 8.0),
            salto_range=(1.7, 2.3),
            abdominais_range=(28, 45),
            cluster='Intermediário'
        )

        iniciante = self.gerar_dados(
            n=40,
            sexo_dist=[0.5, 0.5],
            altura_range=(1.55, 1.78),
            envergadura_range=(1.58, 1.82),
            arremesso_range=(3.5, 6.0),
            salto_range=(1.2, 1.8),
            abdominais_range=(15, 32),
            cluster='Iniciante'
        )
        
        if returnType == 'df':
            df = pd.DataFrame(elite + competitivo + intermediario + iniciante)
            return df
        else:
            return elite + competitivo + intermediario + iniciante
        