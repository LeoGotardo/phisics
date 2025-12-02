import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import pearsonr, f_oneway

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


def carregar_dados_seguros(arquivo_csv: str = 'dataset_athletes.csv') -> pd.DataFrame:
    """
    Carrega dados com tratamento de erros robusto.
    
    Args:
        arquivo_csv: Caminho do arquivo
        
    Returns:
        DataFrame ou None em caso de erro
    """
    try:
        if not os.path.exists(arquivo_csv):
            print(f"❌ Erro: Arquivo {arquivo_csv} não encontrado")
            print("   Execute primeiro: python dataGenerator.py")
            return None
        
        df = pd.read_csv(arquivo_csv)
        
        # Validar colunas necessárias
        colunas_necessarias = ['nome', 'dataNascimento', 'sexo', 'altura', 
                              'envergadura', 'arremesso', 'saltoHorizontal', 
                              'abdominais', 'cluster']
        
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            print(f"❌ Erro: Colunas faltando no CSV: {', '.join(colunas_faltantes)}")
            return None
        
        # Converter e validar dados
        df['dataNascimento'] = pd.to_datetime(df['dataNascimento'], errors='coerce')
        df['idade'] = (datetime.now() - df['dataNascimento']).dt.days // 365
        df['sexo_encoded'] = df['sexo'].map({'M': 1, 'F': 0})
        
        # Verificar valores nulos
        if df.isnull().any().any():
            print("⚠️  Aviso: Dados com valores nulos detectados")
            print(df.isnull().sum())
            df = df.dropna()
            print(f"   {len(df)} registros válidos após limpeza")
        
        return df
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return None


def analise_pca_segura(df: pd.DataFrame, features: list) -> dict:
    """
    Análise PCA com tratamento de erros.
    
    Args:
        df: DataFrame
        features: Lista de features
        
    Returns:
        Dicionário com resultados ou erro
    """
    try:
        # Verificar se features existem
        features_faltantes = [f for f in features if f not in df.columns]
        if features_faltantes:
            return {
                'erro': f"Features faltando: {', '.join(features_faltantes)}",
                'sucesso': False
            }
        
        X = df[features].values
        
        # Verificar valores inválidos
        if np.isnan(X).any() or np.isinf(X).any():
            return {
                'erro': "Dados contêm valores NaN ou infinitos",
                'sucesso': False
            }
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        variance_explained = pca.explained_variance_ratio_
        
        return {
            'sucesso': True,
            'X_pca': X_pca,
            'pca': pca,
            'variance': variance_explained,
            'scaler': scaler
        }
        
    except Exception as e:
        return {
            'erro': str(e),
            'sucesso': False
        }


def analise_correlacao_segura(df: pd.DataFrame) -> dict:
    """
    Análise de correlação com tratamento de erros.
    
    Args:
        df: DataFrame
        
    Returns:
        Dicionário com resultados
    """
    try:
        if 'arremesso' not in df.columns or 'saltoHorizontal' not in df.columns:
            return {
                'erro': 'Colunas arremesso ou saltoHorizontal não encontradas',
                'sucesso': False
            }
        
        potencia_superior = df['arremesso'].dropna().values
        potencia_inferior = df['saltoHorizontal'].dropna().values
        
        # Garantir mesmo tamanho
        min_len = min(len(potencia_superior), len(potencia_inferior))
        potencia_superior = potencia_superior[:min_len]
        potencia_inferior = potencia_inferior[:min_len]
        
        if len(potencia_superior) < 3:
            return {
                'erro': 'Dados insuficientes para correlação',
                'sucesso': False
            }
        
        r_value, p_value = pearsonr(potencia_superior, potencia_inferior)
        
        return {
            'sucesso': True,
            'r_value': r_value,
            'p_value': p_value,
            'superior': potencia_superior,
            'inferior': potencia_inferior
        }
        
    except Exception as e:
        return {
            'erro': str(e),
            'sucesso': False
        }


def analise_core_segura(df: pd.DataFrame) -> dict:
    """
    Análise de força do core com tratamento de erros.
    
    Args:
        df: DataFrame
        
    Returns:
        Dicionário com resultados
    """
    try:
        if 'cluster' not in df.columns or 'abdominais' not in df.columns:
            return {
                'erro': 'Colunas cluster ou abdominais não encontradas',
                'sucesso': False
            }
        
        clusters_ordem = ['Iniciante', 'Intermediário', 'Competitivo', 'Elite']
        
        stats = []
        grupos = []
        
        for cluster in clusters_ordem:
            dados_cluster = df[df['cluster'] == cluster]['abdominais'].dropna()
            
            if len(dados_cluster) == 0:
                continue
            
            grupos.append(dados_cluster.values)
            
            stats.append({
                'cluster': cluster,
                'media': float(dados_cluster.mean()),
                'std': float(dados_cluster.std()),
                'min': int(dados_cluster.min()),
                'max': int(dados_cluster.max()),
                'count': len(dados_cluster)
            })
        
        if len(grupos) < 2:
            return {
                'erro': 'Dados insuficientes para ANOVA (menos de 2 grupos)',
                'sucesso': False
            }
        
        # ANOVA
        f_stat, p_value = f_oneway(*grupos)
        
        return {
            'sucesso': True,
            'stats': stats,
            'f_stat': float(f_stat),
            'p_value': float(p_value),
            'grupos': grupos
        }
        
    except Exception as e:
        return {
            'erro': str(e),
            'sucesso': False
        }


def calcular_metricas_seguras(df: pd.DataFrame, features: list) -> dict:
    """
    Calcula métricas com tratamento de erros.
    
    Args:
        df: DataFrame
        features: Lista de features
        
    Returns:
        Dicionário com métricas
    """
    try:
        # Verificar features
        features_faltantes = [f for f in features if f not in df.columns]
        if features_faltantes:
            return {
                'erro': f"Features faltando: {', '.join(features_faltantes)}",
                'sucesso': False
            }
        
        X = df[features].values
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Converter clusters
        cluster_mapping = {
            'Iniciante': 0,
            'Intermediário': 1,
            'Competitivo': 2,
            'Elite': 3
        }
        
        if 'cluster' not in df.columns:
            return {
                'erro': 'Coluna cluster não encontrada',
                'sucesso': False
            }
        
        labels = df['cluster'].map(cluster_mapping).values
        
        # Verificar se há clusters válidos
        if np.isnan(labels).any():
            return {
                'erro': 'Clusters com valores inválidos',
                'sucesso': False
            }
        
        # Calcular métricas
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        # Calcular inércia
        inertia = 0
        for cluster_id in range(4):
            cluster_data = X_scaled[labels == cluster_id]
            if len(cluster_data) > 0:
                centroid = cluster_data.mean(axis=0)
                inertia += np.sum((cluster_data - centroid) ** 2)
        
        return {
            'sucesso': True,
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin),
            'inertia': float(inertia)
        }
        
    except Exception as e:
        return {
            'erro': str(e),
            'sucesso': False
        }


def diagnostico_completo(arquivo_csv: str = 'dataset_athletes.csv'):
    """
    Executa diagnóstico completo do dataset.
    
    Args:
        arquivo_csv: Caminho do arquivo
    """
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO COMPLETO DO DATASET")
    print("=" * 70)
    
    # 1. Carregar dados
    print("\n1. Carregando dados...")
    df = carregar_dados_seguros(arquivo_csv)
    
    if df is None:
        print("\n❌ Não foi possível carregar os dados")
        return
    
    print(f"✓ {len(df)} registros carregados com sucesso")
    
    # 2. Informações gerais
    print("\n2. Informações Gerais:")
    print(f"   - Total de atletas: {len(df)}")
    print(f"   - Colunas: {', '.join(df.columns)}")
    print(f"   - Tipos de dados:\n{df.dtypes}")
    
    # 3. Distribuição
    print("\n3. Distribuição de Clusters:")
    print(df['cluster'].value_counts())
    
    print("\n4. Distribuição de Sexo:")
    print(df['sexo'].value_counts())
    
    # 5. Estatísticas descritivas
    print("\n5. Estatísticas Descritivas:")
    print(df[['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']].describe())
    
    # 6. Análises
    features = ['sexo_encoded', 'altura', 'envergadura', 
               'arremesso', 'saltoHorizontal', 'abdominais']
    
    print("\n6. Análise PCA:")
    resultado_pca = analise_pca_segura(df, features)
    if resultado_pca['sucesso']:
        var_exp = resultado_pca['variance']
        print(f"   ✓ PC1: {var_exp[0]*100:.1f}% variância")
        print(f"   ✓ PC2: {var_exp[1]*100:.1f}% variância")
        print(f"   ✓ Total: {sum(var_exp)*100:.1f}%")
    else:
        print(f"   ❌ Erro: {resultado_pca['erro']}")
    
    print("\n7. Análise de Correlação:")
    resultado_corr = analise_correlacao_segura(df)
    if resultado_corr['sucesso']:
        print(f"   ✓ Correlação: r = {resultado_corr['r_value']:.3f}")
        print(f"   ✓ P-value: {resultado_corr['p_value']:.6f}")
    else:
        print(f"   ❌ Erro: {resultado_corr['erro']}")
    
    print("\n8. Análise de Core:")
    resultado_core = analise_core_segura(df)
    if resultado_core['sucesso']:
        print(f"   ✓ ANOVA: F = {resultado_core['f_stat']:.2f}, p = {resultado_core['p_value']:.6f}")
        for stat in resultado_core['stats']:
            print(f"   - {stat['cluster']}: {stat['media']:.1f} ± {stat['std']:.1f} rep/min")
    else:
        print(f"   ❌ Erro: {resultado_core['erro']}")
    
    print("\n9. Métricas de Clustering:")
    resultado_metricas = calcular_metricas_seguras(df, features)
    if resultado_metricas['sucesso']:
        print(f"   ✓ Silhouette: {resultado_metricas['silhouette']:.3f}")
        print(f"   ✓ Davies-Bouldin: {resultado_metricas['davies_bouldin']:.3f}")
        print(f"   ✓ Inércia: {resultado_metricas['inertia']:.0f}")
    else:
        print(f"   ❌ Erro: {resultado_metricas['erro']}")
    
    # 10. Verificar problemas
    print("\n10. Verificação de Problemas:")
    problemas = []
    
    # Valores nulos
    nulos = df.isnull().sum()
    if nulos.any():
        problemas.append(f"Valores nulos encontrados:\n{nulos[nulos > 0]}")
    
    # Valores duplicados
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        problemas.append(f"{duplicados} registros duplicados")
    
    # Outliers
    for col in ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            problemas.append(f"{col}: {outliers} possíveis outliers")
    
    if problemas:
        print("   ⚠️  Problemas encontrados:")
        for p in problemas:
            print(f"   - {p}")
    else:
        print("   ✓ Nenhum problema detectado")
    
    print("\n" + "=" * 70)
    print("✅ DIAGNÓSTICO CONCLUÍDO")
    print("=" * 70 + "\n")


def main():
    """Função principal"""
    diagnostico_completo()