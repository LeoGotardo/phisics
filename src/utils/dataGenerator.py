import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataGenerator:
    """
    Gerador de dados sintéticos para treinamento do modelo de classificação de atletas.
    Gera dados realistas com distribuições adequadas para cada cluster.
    """
    
    def __init__(self, nAthletes: int = 160) -> None:
        self.nAthletes = nAthletes
        self.nomes_masculinos = [
            'João', 'Pedro', 'Lucas', 'Gabriel', 'Matheus', 'Rafael', 'Bruno', 
            'Felipe', 'Guilherme', 'Rodrigo', 'Diego', 'Fernando', 'André', 
            'Carlos', 'Eduardo', 'Thiago', 'Marcelo', 'Daniel', 'Leonardo', 'Paulo'
        ]
        self.nomes_femininos = [
            'Maria', 'Ana', 'Juliana', 'Fernanda', 'Carla', 'Beatriz', 'Camila',
            'Amanda', 'Paula', 'Larissa', 'Mariana', 'Patrícia', 'Renata', 
            'Gabriela', 'Carolina', 'Vanessa', 'Bianca', 'Letícia', 'Natália', 'Débora'
        ]
        self.sobrenomes = [
            'Silva', 'Santos', 'Oliveira', 'Souza', 'Costa', 'Ferreira', 'Rodrigues',
            'Alves', 'Pereira', 'Lima', 'Gomes', 'Martins', 'Ribeiro', 'Carvalho',
            'Rocha', 'Almeida', 'Nascimento', 'Araújo', 'Fernandes', 'Barbosa'
        ]
    
    def gerar_nome(self, sexo: str) -> str:
        """Gera um nome aleatório baseado no sexo"""
        if sexo == 'M':
            primeiro_nome = random.choice(self.nomes_masculinos)
        else:
            primeiro_nome = random.choice(self.nomes_femininos)
        
        sobrenome = random.choice(self.sobrenomes)
        return f"{primeiro_nome} {sobrenome}"
    
    
    def gerar_data_nascimento(self) -> datetime:
        """Gera uma data de nascimento aleatória entre 15 e 35 anos atrás"""
        hoje = datetime.now()
        anos_atras = random.randint(15, 35)
        dias_random = random.randint(0, 365)
        return hoje - timedelta(days=anos_atras * 365 + dias_random)
    
    
    def gerar_dados_cluster(self, n: int, sexo_dist: list, altura_range: tuple, 
                           envergadura_range: tuple, arremesso_range: tuple, 
                           salto_range: tuple, abdominais_range: tuple, 
                           cluster: str) -> list:
        """
        Gera dados para um cluster específico com distribuições realistas.
        
        Args:
            n: Número de atletas a gerar
            sexo_dist: [prob_masculino, prob_feminino]
            altura_range: (min, max) altura em metros
            envergadura_range: (min, max) envergadura em metros
            arremesso_range: (min, max) arremesso em metros
            salto_range: (min, max) salto horizontal em metros
            abdominais_range: (min, max) abdominais em repetições
            cluster: Nome do cluster
            
        Returns:
            Lista de dicionários com dados dos atletas
        """
        dados = []
        
        for _ in range(n):
            sexo = np.random.choice(['M', 'F'], p=sexo_dist)
            nome = self.gerar_nome(sexo)
            data_nascimento = self.gerar_data_nascimento()
            
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
                abdominais = np.random.randint(
                    int(abdominais_range[0] * 0.85), 
                    int(abdominais_range[1] * 0.9)
                )
            
            dados.append({
                'nome': nome,
                'dataNascimento': data_nascimento.strftime('%Y-%m-%d'),
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
        """
        Gera o dataset completo com todos os clusters.
        
        Args:
            returnType: 'df' para DataFrame, 'list' para lista de dicts, 'csv' para salvar em arquivo
            
        Returns:
            DataFrame ou lista dependendo de returnType
        """
        elite = self.gerar_dados_cluster(
            n=int(self.nAthletes * 0.25),
            sexo_dist=[0.6, 0.4],
            altura_range=(1.65, 1.85),
            envergadura_range=(1.70, 1.95),
            arremesso_range=(10, 14),
            salto_range=(2.6, 3.2),
            abdominais_range=(55, 75),
            cluster='Elite'
        )

        competitivo = self.gerar_dados_cluster(
            n=int(self.nAthletes * 0.25),
            sexo_dist=[0.55, 0.45],
            altura_range=(1.60, 1.82),
            envergadura_range=(1.65, 1.90),
            arremesso_range=(7.5, 10.5),
            salto_range=(2.1, 2.7),
            abdominais_range=(40, 58),
            cluster='Competitivo'
        )

        intermediario = self.gerar_dados_cluster(
            n=int(self.nAthletes * 0.25),
            sexo_dist=[0.5, 0.5],
            altura_range=(1.58, 1.80),
            envergadura_range=(1.60, 1.85),
            arremesso_range=(5.5, 8.0),
            salto_range=(1.7, 2.3),
            abdominais_range=(28, 45),
            cluster='Intermediário'
        )

        iniciante = self.gerar_dados_cluster(
            n=int(self.nAthletes * 0.25),
            sexo_dist=[0.5, 0.5],
            altura_range=(1.55, 1.78),
            envergadura_range=(1.58, 1.82),
            arremesso_range=(3.5, 6.0),
            salto_range=(1.2, 1.8),
            abdominais_range=(15, 32),
            cluster='Iniciante'
        )
        
        # Combinar todos os dados
        todos_dados = elite + competitivo + intermediario + iniciante
        
        if returnType == 'list':
            return todos_dados
        
        df = pd.DataFrame(todos_dados)
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if returnType == 'csv':
            df.to_csv('dataset_athletes.csv', index=False)
            
            print("=" * 70)
            print("DATASET CRIADO COM SUCESSO!")
            print("=" * 70)
            print(f"\nTotal de exemplos: {len(df)}")
            print("\nDistribuição por cluster:")
            print(df['cluster'].value_counts().sort_index())
            print("\nDistribuição por sexo:")
            print(df['sexo'].value_counts())
            print("\nPrimeiras linhas do dataset:")
            print(df.head(10))
            print("\nEstatísticas descritivas por cluster:")
            print(df.groupby('cluster')[['altura', 'envergadura', 'arremesso', 
                                         'saltoHorizontal', 'abdominais']].mean().round(2))
            print("\n✓ Arquivo salvo: dataset_athletes.csv")
            print("=" * 70)
            
            # Salvar também arquivos separados por cluster
            for cluster in ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']:
                df_cluster = df[df['cluster'] == cluster]
                filename = f'dataset_{cluster.lower()}.csv'
                df_cluster.to_csv(filename, index=False)
                print(f"✓ Arquivo salvo: {filename}")
            
            return df
        
        return df