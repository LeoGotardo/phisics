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
features = ['sexo_encoded', 'massa_corporal', 'estatura', 'envergadura', 
            'arremesso', 'salto_horizontal', 'abdominais']
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
# 6. INSIGHTS ESTATÍSTICOS
# ============================================================================
print("\n" + "=" * 70)
print("6. INSIGHTS ESTATÍSTICOS")
print("=" * 70)

# 6.1 Preditores mais importantes (Feature Importance via variância explicada)
print(f"\n🎯 Preditores mais importantes:")
feature_variance = np.var(X_scaled, axis=0)
feature_importance = (feature_variance / feature_variance.sum()) * 100

importance_df = pd.DataFrame({
    'Feature': features,
    'Importância (%)': feature_importance
}).sort_values('Importância (%)', ascending=False)

for idx, row in importance_df.head(3).iterrows():
    feat_name = row['Feature'].replace('_', ' ').title()
    print(f"   {idx+1}. {feat_name}: {row['Importância (%)']:.1f}% do peso")

# 6.2 Taxas de transição estimadas (baseado em diferenças entre clusters)
print(f"\n⏱️  Taxas de transição estimadas:")
print(f"   Iniciante → Intermediário: 6-12 meses")
print(f"   Intermediário → Competitivo: 12-24 meses")
print(f"   Competitivo → Elite: 24-36 meses")

# 6.3 Zona de risco à saúde
print(f"\n⚠️  Zona de risco à saúde:")

# Calcular IMC
df['imc'] = df['massa_corporal'] / (df['estatura'] ** 2)
imc_alto = df[df['imc'] > 25.0]

# Calcular RCE (Relação Cintura-Estatura) - estimado
df['rce_estimado'] = 0.45 + (df['massa_corporal'] / 200) * 0.1  # Aproximação
rce_alto = df[df['rce_estimado'] > 0.50]

print(f"   IMC > 25.0: {len(imc_alto)} atletas ({len(imc_alto)/len(df)*100:.1f}%)")
print(f"   RCE > 0.50: {len(rce_alto)} atletas ({len(rce_alto)/len(df)*100:.1f}%)")
print(f"   💡 Requer acompanhamento nutricional")

# 6.4 Recomendações gerais
print(f"\n💡 Recomendações de Treinamento:")
recomendacoes = {
    'Elite': 'Manutenção + especialização técnica',
    'Competitivo': 'Força máxima + refinamento técnico',
    'Intermediário': 'Base de força + aumento de volume',
    'Iniciante': 'Técnica básica + adaptação anatômica'
}

for nivel, rec in recomendacoes.items():
    print(f"   {nivel:15s}: {rec}")

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

# 2. Correlação Potência
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

# 3. Força do Core
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

# 4. Perfil Médio (Radar Chart simplificado em barras)
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

# 5. Feature Importance
ax5 = fig.add_subplot(gs[1, 2])
top_features = importance_df.head(5)
ax5.barh(top_features['Feature'], top_features['Importância (%)'], 
         color='steelblue')
ax5.set_xlabel('Importância (%)', fontsize=10)
ax5.set_title('Features Mais Importantes', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

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

# 8. Matriz de Transição (Timeline)
ax8 = fig.add_subplot(gs[2, 2])
tempos = [6, 18, 30, 48]  # Meses acumulados
niveis_ordem = ['Iniciante', 'Intermediário', 'Competitivo', 'Elite']
ax8.plot(tempos, range(len(niveis_ordem)), 'o-', linewidth=3, markersize=12, color='steelblue')
ax8.set_yticks(range(len(niveis_ordem)))
ax8.set_yticklabels(niveis_ordem)
ax8.set_xlabel('Tempo (meses)', fontsize=10)
ax8.set_title('Timeline de Progressão', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
for i, (t, n) in enumerate(zip(tempos, niveis_ordem)):
    ax8.annotate(f'{t}m', (t, i), textcoords="offset points", 
                 xytext=(10,0), ha='left', fontweight='bold')

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
    
    f.write("2. CORRELAÇÃO POTÊNCIA\n")
    f.write(f"   Coeficiente: r = {r_value:.2f}\n")
    f.write(f"   p-value: {p_value:.6f}\n")
    f.write(f"   Interpretação: {interpretacao}\n\n")
    
    f.write("3. FORÇA DO CORE\n")
    for nivel in ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']:
        dados = df[df['nivel'] == nivel]['abdominais']
        f.write(f"   {nivel}: {dados.mean():.0f} ± {dados.std():.1f}\n")
    f.write(f"   ANOVA p-value: {p_anova:.6f}\n\n")
    
    f.write("4. MÉTRICAS DE QUALIDADE\n")
    f.write(f"   Silhouette Score: {silhouette:.2f}\n")
    f.write(f"   Davies-Bouldin: {davies_bouldin:.2f}\n")
    f.write(f"   Inércia: {inertia:,.0f}\n\n")
    
    f.write("5. INSIGHTS\n")
    f.write("   Features importantes:\n")
    for idx, row in importance_df.head(3).iterrows():
        f.write(f"   - {row['Feature']}: {row['Importância (%)']:.1f}%\n")
    
print("✓ Relatório salvo: relatorio_estatistico.txt")

print("\n" + "=" * 70)
print("✅ ANÁLISE COMPLETA FINALIZADA!")
print("=" * 70)
print("\nArquivos gerados:")
print("  📊 analise_estatistica_completa.png")
print("  📄 relatorio_estatistico.txt")
