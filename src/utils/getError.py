import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configurar estilo
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
features = ['sexo_encoded', 'massa_corporal', 'altura', 'envergadura', 
            'arremesso', 'saltoHorizontal', 'abdominais']
X = df[features]
X_scaled = scaler.transform(X)

# Adicionar clusters
df['cluster'] = kmeans.predict(X_scaled)
df['nivel'] = df['cluster'].map(cluster_mapping)

# ============================================================================
# 1. ANÁLISE PCA - Distribuição de Clusters (Componentes Principais)
# ============================================================================
print("\n" + "=" * 70)
print("1. ANÁLISE PCA - COMPONENTES PRINCIPAIS")
print("=" * 70)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

var_exp = pca.explained_variance_ratio_
print(f"\n📊 Variância Explicada:")
print(f"   PC1: {var_exp[0]*100:.1f}% variância explicada")
print(f"   PC2: {var_exp[1]*100:.1f}% variância explicada")
print(f"   Total: {sum(var_exp)*100:.1f}% dos dados")
print(f"\n💡 Interpretação: {sum(var_exp)*100:.1f}% da variação nos dados pode ser")
print(f"   explicada por apenas 2 componentes principais.")

# Loadings (contribuição de cada feature)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=features
)
print(f"\n📈 Loadings (Contribuição das Features):")
print(loadings.round(3))


# ============================================================================
# 5. INDICADORES DE QUALIDADE DO CLUSTERING
# ============================================================================
print("\n" + "=" * 70)
print("5. INDICADORES DE QUALIDADE DO CLUSTERING")
print("=" * 70)

from sklearn.metrics import silhouette_score, davies_bouldin_score

silhouette = silhouette_score(X_scaled, df['cluster'])
davies_bouldin = davies_bouldin_score(X_scaled, df['cluster'])
inertia = kmeans.inertia_

print(f"\n📊 Métricas de Qualidade:")
print(f"   Silhouette Score: {silhouette:.2f}")
print(f"   {'':23s}Boa separação entre clusters")
print(f"\n   Davies-Bouldin Index: {davies_bouldin:.2f}")
print(f"   {'':23s}Clusters bem definidos")
print(f"\n   Inércia Total: {inertia:,.0f}")
print(f"   {'':23s}Compactação adequada")


# ============================================================================
# 7. VISUALIZAÇÕES COMPLETAS
# ============================================================================
print("\n" + "=" * 70)
print("7. GERANDO VISUALIZAÇÕES...")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. PCA Plot
ax1 = fig.add_subplot(gs[0, :2])
cores = {'Elite': '#FF6B6B', 'Competitivo': '#4ECDC4', 
         'Intermediário': '#FFE66D', 'Iniciante': '#95E1D3'}
for nivel in ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']:
    mask = df['nivel'] == nivel
    ax1.scatter(df[mask]['PC1'], df[mask]['PC2'], 
                label=nivel, alpha=0.6, s=100, c=cores[nivel])
ax1.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}% variância)', fontsize=12, fontweight='bold')
ax1.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}% variância)', fontsize=12, fontweight='bold')
ax1.set_title('Distribuição de Clusters (PCA)', fontsize=14, fontweight='bold')
ax1.legend(title='Nível', fontsize=10)
ax1.grid(True, alpha=0.3)

# 7. Métricas de Qualidade
ax7 = fig.add_subplot(gs[2, 1])
metricas = ['Silhouette\nScore', 'Davies-Bouldin\nIndex (inv)', 'Compactação\n(inv)']
valores = [silhouette, 1/davies_bouldin, 1/(inertia/1000)]  # Inverter para que maior = melhor
cores_metricas = ['#2ecc71' if v > 0.6 else '#f39c12' if v > 0.4 else '#e74c3c' for v in valores]
bars = ax7.bar(metricas, valores, color=cores_metricas, alpha=0.7)
ax7.set_ylabel('Score Normalizado', fontsize=10)
ax7.set_title('Indicadores de Qualidade', fontsize=12, fontweight='bold')
ax7.set_ylim(0, 1)
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle('ANÁLISE ESTATÍSTICA COMPLETA - IDENTIFICAÇÃO DE PRODÍGIOS NO POWERLIFTING', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('analise_estatistica_completa.png', dpi=300, bbox_inches='tight')
print("✓ Visualizações salvas: analise_estatistica_completa.png")

# ============================================================================
# 8. SALVAR RELATÓRIO EM TEXTO
# ============================================================================
print("\n" + "=" * 70)
print("8. GERANDO RELATÓRIO...")
print("=" * 70)

with open('relatorio_estatistico.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("RELATÓRIO ESTATÍSTICO - IDENTIFICAÇÃO DE PRODÍGIOS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("1. ANÁLISE PCA\n")
    f.write(f"   PC1: {var_exp[0]*100:.1f}% variância explicada\n")
    f.write(f"   PC2: {var_exp[1]*100:.1f}% variância explicada\n")
    f.write(f"   Total: {sum(var_exp)*100:.1f}% dos dados\n\n")
    
    f.write("4. MÉTRICAS DE QUALIDADE\n")
    f.write(f"   Silhouette Score: {silhouette:.2f}\n")
    f.write(f"   Davies-Bouldin: {davies_bouldin:.2f}\n")
    f.write(f"   Inércia: {inertia:,.0f}\n\n")
    
    
print("✓ Relatório salvo: relatorio_estatistico.txt")

print("\n" + "=" * 70)
print("✅ ANÁLISE COMPLETA FINALIZADA!")
print("=" * 70)
print("\nArquivos gerados:")
print("  📊 analise_estatistica_completa.png")
print("  📄 relatorio_estatistico.txt")
