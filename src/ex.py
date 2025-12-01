#Exemplo

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from services.knn_maps import Volei, Futebol, Basquete, Lutas, Natacao, Handebol

def tipo_esporte(classe_esporte, nome_esporte):
    esporte_data = {
        'Idade': classe_esporte.list_idade(),
        'Altura': classe_esporte.list_altura(),
        'Peso': classe_esporte.list_peso(),
        'Flexibilidade': classe_esporte.list_flexibilidade(),
        'Salto Horizontal': classe_esporte.list_horizontal(),
        'Salto Vertical': classe_esporte.list_vertical(),
        'Arremesso': classe_esporte.list_arremesso(),
        'Resistência': classe_esporte.list_resistencia(),
    }
    df = pd.DataFrame(esporte_data)
    df['Esporte'] = nome_esporte
    return df

lista_dataframes_esportes = [
    tipo_esporte(Futebol, "Futebol"),
    tipo_esporte(Volei, "Vôlei"),
    tipo_esporte(Basquete, "Basquete"),
    tipo_esporte(Lutas, "Lutas"),
    tipo_esporte(Natacao, "Natação"),
    tipo_esporte(Handebol, "Handebol")
]

df = pd.concat(lista_dataframes_esportes, ignore_index=True)

print("--- Verificando as colunas do DataFrame ---")
print(df.columns)
print("-" * 40)

# Agora, a linha que está dando erro
X_train = df.drop('Esporte', axis=1) # <-- Mudei para 'Esporte' com 'E' maiúsculo
y_train_labels = df['Esporte']      # <-- Mudei aqui também

print("--- 1. Base de Dados para Treinamento ---")
print(df)
print("-" * 40)

# 2. SEUS DADOS PARA CLASSIFICAÇÃO
# A ordem das colunas deve ser EXATAMENTE a mesma do DataFrame acima (exceto 'esporte')

marcelo = np.array([[
    20,
    182, # altura_cm
    77.6,  # peso_kg
    42,    # flexibilidade_cm
    197,   # saltoHorizontal_cm
    45.8,   # salto_vertical_cm
    5.6,   # arremesso_mb_m
    52    # abdominais_1min
]])

sla = np.array([[
    25,
    173, # altura_cm
    73.1,  # peso_kg
    40,    # flexibilidade_cm
    83,   # saltoHorizontal_cm
    21.1,   # salto_vertical_cm
    4.2,   # arremesso_mb_m
    25    # abdominais_1min
]])

eu = np.array([[
    24,
    161.9, # altura_cm
    54.1,  # peso_kg
    45,    # flexibilidade_cm
    122,   # saltoHorizontal_cm
    21.9,   # salto_vertical_cm
    3.4,   # arremesso_mb_m
    26    # abdominais_1min
]])

# --- O PROCESSO DO MODELO COMEÇA AQUI ---

# 3. PREPARAÇÃO DOS DADOS
# Separando as features (X) da variável alvo (y) do nosso dataset de treino
X_train = df.drop('Esporte', axis=1)
y_train_labels = df['Esporte']

# a) Convertendo os nomes dos esportes (texto) para números
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_labels)

# b) Padronizando as features numéricas
# Criamos o scaler com os dados de treino
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

def info_dados():
    list = []
    list.append(len(Futebol.list_idade()))
    list.append(len(Volei.list_idade()))
    list.append(len(Basquete.list_idade()))
    list.append(len(Lutas.list_idade()))
    list.append(len(Natacao.list_idade()))
    list.append(len(Handebol.list_idade()))
    return list

k = len(info_dados())
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train_encoded)

print("--- 2. Modelo treinado com sucesso! ---")
print("-" * 40)


# 5. CLASSIFICANDO SEU PERFIL
# IMPORTANTE: Seus dados devem ser transformados com o MESMO scaler treinado acima
meus_dados_scaled = scaler.transform(sla)

# Fazendo a previsão
previsao_numerica = knn.predict(meus_dados_scaled)
probabilidades = knn.predict_proba(meus_dados_scaled)

# Convertendo o resultado numérico de volta para o nome do esporte
previsao_esporte = label_encoder.inverse_transform(previsao_numerica)

print("--- 3. Resultado da Classificação para 'Tiago' ---")
print(f"Seu perfil físico é mais similar aos atletas de: {previsao_esporte[0]}")
print("\nProbabilidade de similaridade com cada esporte:")

# Mostra a 'confiança' do modelo para cada classe
for i, sport in enumerate(label_encoder.classes_):
    print(f"- {sport}: {probabilidades[0][i]*100:.2f}%")
print("-" * 40)