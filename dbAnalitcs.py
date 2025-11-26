import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("=" * 70)
print("ANÁLISE ESTATÍSTICA AVANÇADA - IDENTIFICAÇÃO DE PRODÍGIOS")
print("=" * 70)

# 1. CARREGAR DADOS E MODELO
df = pd.read_csv('dataset_powerlifting_clusters.csv')
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
cluster_mapping = joblib.load('cluster_mapping.pkl')

# Preparar dados
df['sexo_encoded'] = df['sexo'].map({'M': 1, 'F': 0})
features = ['sexo_encoded', 'massa_corporal', 'estatura', 'envergadura', 
            'arremesso', 'salto_horizontal', 'abdominais']
X = df[features]
X_scaled = scaler.transform(X)

# Adicionar clusters
df['cluster'] = kmeans.predict(X_scaled)
df['nivel'] = df['cluster'].map(cluster_mapping)


# ============================================================================
# 2. CORRELAÇÃO - Potência Superior vs Inferior
# ============================================================================
print("\n" + "=" * 70)
print("2. CORRELAÇÃO - POTÊNCIA SUPERIOR VS INFERIOR")
print("=" * 70)

# Criar variáveis de potência
df['potencia_superior'] = df['arremesso']  # Arremesso representa força superior
df['potencia_inferior'] = df['salto_horizontal']  # Salto representa força inferior

correlation = stats.pearsonr(df['potencia_superior'], df['potencia_inferior'])
r_value = correlation[0]
p_value = correlation[1]

print(f"\n🔗 Coeficiente de Correlação de Pearson:")
print(f"   r = {r_value:.2f}")
print(f"   p-value = {p_value:.6f}")

if p_value < 0.001:
    sig = "p < 0.001 - Altamente significativo"
elif p_value < 0.05:
    sig = f"p = {p_value:.3f} - Significativo"
else:
    sig = f"p = {p_value:.3f} - Não significativo"

if abs(r_value) > 0.7:
    interpretacao = "Correlação forte positiva"
elif abs(r_value) > 0.4:
    interpretacao = "Correlação moderada positiva"
else:
    interpretacao = "Correlação fraca"

print(f"   Significância: {sig}")
print(f"   Interpretação: {interpretacao}")
print(f"\n💡 Atletas com alta potência superior tendem a ter alta potência inferior.")
print(f"   Isso indica desenvolvimento equilibrado do corpo.")

# ============================================================================
# 3. FORÇA DO CORE POR CLUSTER (ANOVA)
# ============================================================================
print("\n" + "=" * 70)
print("3. FORÇA DO CORE POR CLUSTER")
print("=" * 70)

# Estatísticas descritivas por cluster
print(f"\n💪 Abdominais (rep/min) por Nível:")
for nivel in ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']:
    dados = df[df['nivel'] == nivel]['abdominais']
    media = dados.mean()
    desvio = dados.std()
    print(f"   {nivel:15s}: {media:.0f} ± {desvio:.1f}")

# ANOVA - Teste de diferença entre grupos
grupos = [df[df['nivel'] == nivel]['abdominais'].values 
          for nivel in ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']]
f_stat, p_anova = f_oneway(*grupos)

print(f"\n📊 ANOVA (Analysis of Variance):")
print(f"   F-statistic: {f_stat:.2f}")
print(f"   p-value: {p_anova:.6f}")
if p_anova < 0.001:
    print(f"   Resultado: Diferença SIGNIFICATIVA entre clusters (p < 0.001)")
    print(f"   💡 Core forte distingue atletas de elite dos demais níveis.")
else:
    print(f"   Resultado: Sem diferença significativa entre clusters")
    


# ============================================================================
# 4. PERFIL MÉDIO POR CLUSTER (Scores Normalizados 0-100)
# ============================================================================
print("\n" + "=" * 70)
print("4. PERFIL MÉDIO POR CLUSTER")
print("=" * 70)

# Normalizar cada feature para escala 0-100
features_fisicas = ['arremesso', 'salto_horizontal', 'abdominais']
df_normalized = df.copy()

for feat in features_fisicas:
    min_val = df[feat].min()
    max_val = df[feat].max()
    df_normalized[f'{feat}_norm'] = ((df[feat] - min_val) / (max_val - min_val)) * 100

print(f"\n📈 Scores Normalizados (0-100):")
for nivel in ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']:
    dados_nivel = df_normalized[df_normalized['nivel'] == nivel]
    scores = []
    for feat in features_fisicas:
        score = dados_nivel[f'{feat}_norm'].mean()
        scores.append(score)
    
    min_score = min(scores)
    max_score = max(scores)
    print(f"   {nivel:15s}: {min_score:.0f}-{max_score:.0f} em todas dimensões")
    
    # Calcular coeficiente de variação (homogeneidade)
    cv = (dados_nivel[[f'{f}_norm' for f in features_fisicas]].std().mean() / 
          dados_nivel[[f'{f}_norm' for f in features_fisicas]].mean().mean())
    
    if cv < 0.2:
        print(f"   {'':15s}  → Desenvolvimento homogêneo (CV={cv:.2f})")
    else:
        print(f"   {'':15s}  → Desenvolvimento desigual (CV={cv:.2f})")
        
        
print("\n" + "=" * 70)
print("7. GERANDO VISUALIZAÇÕES...")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

cores = {'Elite': '#FF6B6B', 'Competitivo': '#4ECDC4', 
         'Intermediário': '#FFE66D', 'Iniciante': '#95E1D3'}

# 1. Correlação Potência
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(df['potencia_superior'], df['potencia_inferior'], 
            alpha=0.5, c=df['cluster'], cmap='viridis')
ax2.set_xlabel('Potência Superior (Arremesso)', fontsize=10)
ax2.set_ylabel('Potência Inferior (Salto)', fontsize=10)
ax2.set_title(f'Correlação: r={r_value:.2f}', fontsize=12, fontweight='bold')
z = np.polyfit(df['potencia_superior'], df['potencia_inferior'], 1)
p = np.poly1d(z)
ax2.plot(df['potencia_superior'], p(df['potencia_superior']), 
         "r--", alpha=0.8, linewidth=2)
ax2.grid(True, alpha=0.3)

# 2. Força do Core
ax3 = fig.add_subplot(gs[1, 0])
data_core = [df[df['nivel'] == nivel]['abdominais'].values 
             for nivel in ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']]
bp = ax3.boxplot(data_core, labels=['Elite', 'Comp.', 'Inter.', 'Inic.'],
                  patch_artist=True)
for patch, cor in zip(bp['boxes'], cores.values()):
    patch.set_facecolor(cor)
ax3.set_ylabel('Abdominais (rep/min)', fontsize=10)
ax3.set_title('Força do Core por Cluster', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 3. Perfil Médio (Radar Chart simplificado em barras)
ax4 = fig.add_subplot(gs[1, 1])
niveis = ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']
scores_medios = []
for nivel in niveis:
    dados_nivel = df_normalized[df_normalized['nivel'] == nivel]
    score_medio = dados_nivel[[f'{f}_norm' for f in features_fisicas]].mean().mean()
    scores_medios.append(score_medio)
bars = ax4.barh(niveis, scores_medios, color=list(cores.values()))
ax4.set_xlabel('Score Normalizado (0-100)', fontsize=10)
ax4.set_title('Perfil Médio por Cluster', fontsize=12, fontweight='bold')
ax4.set_xlim(0, 100)
for i, v in enumerate(scores_medios):
    ax4.text(v + 2, i, f'{v:.0f}', va='center', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')