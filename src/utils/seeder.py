import pandas as pd
import numpy as np


def generateData(returnType: str = 'df') -> pd.DataFrame | list:
    # Função para gerar dados com distribuição realista
    def gerar_dados(n, sexo_dist, altura_range, envergadura_range, 
                    arremesso_range, salto_range, abdominais_range, cluster):
        dados = []
        
        for _ in range(n):
            # Distribuição de sexo
            sexo = np.random.choice(['M', 'F'], p=sexo_dist)
            
            # Ajustes baseados no sexo
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

    # CLUSTER ELITE - Atletas de alto nível
    elite = gerar_dados(
        n=40,
        sexo_dist=[0.6, 0.4],  # 60% M, 40% F
        altura_range=(1.65, 1.85),
        envergadura_range=(1.70, 1.95),
        arremesso_range=(10, 14),
        salto_range=(2.6, 3.2),
        abdominais_range=(55, 75),
        cluster='Elite'
    )

    # CLUSTER COMPETITIVO - Atletas experientes
    competitivo = gerar_dados(
        n=40,
        sexo_dist=[0.55, 0.45],
        altura_range=(1.60, 1.82),
        envergadura_range=(1.65, 1.90),
        arremesso_range=(7.5, 10.5),
        salto_range=(2.1, 2.7),
        abdominais_range=(40, 58),
        cluster='Competitivo'
    )

    # CLUSTER INTERMEDIÁRIO - Praticantes regulares
    intermediario = gerar_dados(
        n=40,
        sexo_dist=[0.5, 0.5],
        altura_range=(1.58, 1.80),
        envergadura_range=(1.60, 1.85),
        arremesso_range=(5.5, 8.0),
        salto_range=(1.7, 2.3),
        abdominais_range=(28, 45),
        cluster='Intermediário'
    )

    # CLUSTER INICIANTE - Novatos
    iniciante = gerar_dados(
        n=40,
        sexo_dist=[0.5, 0.5],
        altura_range=(1.55, 1.78),
        envergadura_range=(1.58, 1.82),
        arremesso_range=(3.5, 6.0),
        salto_range=(1.2, 1.8),
        abdominais_range=(15, 32),
        cluster='Iniciante'
    )

    # Combinando todos os dados
    todos_dados = elite + competitivo + intermediario + iniciante

    # Criando DataFrame
    df = pd.DataFrame(todos_dados)

    # Embaralhando os dados
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    if returnType == 'csv':
        # Salvando em CSV
        df.to_csv('dataset.csv', index=False)

        # Exibindo informações
        print("Dataset criado com sucesso!")
        print(f"\nTotal de exemplos: {len(df)}")
        print("\nDistribuição por cluster:")
        print(df['cluster'].value_counts().sort_index())
        print("\nDistribuição por sexo:")
        print(df['sexo'].value_counts())
        print("\nPrimeiras linhas do dataset:")
        print(df.head(10))
        print("\nEstatísticas descritivas por cluster:")
        print(df.groupby('cluster')[['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']].mean().round(2))

        # Salvando também em formato separado por cluster para facilitar análise
        for cluster in ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']:
            df_cluster = df[df['cluster'] == cluster]
            df_cluster.to_csv(f'dataset_{cluster.lower()}.csv', index=False)
            print(f"\nArquivo dataset_{cluster.lower()}.csv criado!")
    else:
        return df
        
        
        
if __name__ == '__main__':
    generateData(returnType='csv')