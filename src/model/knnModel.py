import numpy as np, pandas as pd, joblib, os

from sklearn.metrics import silhouette_score, davies_bouldin_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict


class KNNModel:
    """
    Modelo KNN para classificação de atletas em clusters de desempenho.
    Utiliza características físicas e de desempenho para prever o nível do atleta.
    """
    
    def __init__(self, nNeighbors: int = 5):
        """
        Inicializa o modelo KNN.
        
        Args:
            nNeighbors: Número de vizinhos mais próximos a considerar
        """
        self.nNeighbors = nNeighbors
        self.knn = KNeighborsClassifier(n_neighbors=nNeighbors, weights='distance')
        self.scaler = StandardScaler()
        self.clusterMapping = {
            0: 'Iniciante',
            1: 'Intermediário', 
            2: 'Competitivo',
            3: 'Elite'
        }
        self.reverseMapping = {v: k for k, v in self.clusterMapping.items()}
        self.featureNames = ['sexo', 'altura', 'envergadura', 'arremesso', 
                            'saltoHorizontal', 'abdominais']
        self.isTrained = False
        
    
    def prepareFeatures(self, data: pd.DataFrame | List[Dict]) -> np.ndarray:
        """
        Prepara as features para treinamento ou predição.
        
        Args:
            data: DataFrame ou lista de dicionários com dados dos atletas
            
        Returns:
            Array numpy com features preparadas
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Criar cópia para não modificar original
        dfFeatures = data.copy()
        
        # Codificar sexo (M=1, F=0)
        if 'sexo' in dfFeatures.columns:
            dfFeatures['sexo'] = dfFeatures['sexo'].map({'M': 1, 'F': 0})
        
        # Selecionar apenas as features necessárias
        X = dfFeatures[self.featureNames].values
        
        return X
    
    
    def prepareLabels(self, data: pd.DataFrame | List[Dict]) -> np.ndarray:
        """
        Prepara os labels (clusters) para treinamento.
        
        Args:
            data: DataFrame ou lista de dicionários com dados dos atletas
            
        Returns:
            Array numpy com labels codificados
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Converter nomes dos clusters para códigos numéricos
        if 'cluster' in data.columns:
            y = data['cluster'].map(self.reverseMapping).values
        elif 'nivel' in data.columns:
            y = data['nivel'].map(self.reverseMapping).values
        else:
            raise ValueError("Dados devem conter coluna 'cluster' ou 'nivel'")
        
        return y
    
    
    def fit(self, trainData: pd.DataFrame | List[Dict]) -> Tuple[bool, str]:
        """
        Treina o modelo KNN com os dados fornecidos.
        
        Args:
            trainData: Dados de treinamento
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Preparar features e labels
            X = self.prepareFeatures(trainData)
            y = self.prepareLabels(trainData)
            
            print(y)
            
            # Normalizar features
            XScaled = self.scaler.fit_transform(X)
            
            # Treinar modelo
            self.knn.fit(XScaled, y)
            
            # Calcular métricas de qualidade
            yPred = self.knn.predict(XScaled)
            silhouette = silhouette_score(XScaled, yPred)
            daviesBouldin = davies_bouldin_score(XScaled, yPred)
            
            # Cross-validation score
            cvScores = cross_val_score(self.knn, XScaled, y, cv=5)
            
            self.isTrained = True
            
            mensagem = (
                f"Modelo treinado com sucesso!\n"
                f"Amostras de treino: {len(X)}\n"
                f"Silhouette Score: {silhouette:.3f}\n"
                f"Davies-Bouldin Index: {daviesBouldin:.3f}\n"
                f"Acurácia CV (5-fold): {cvScores.mean():.3f} ± {cvScores.std():.3f}"
            )
            
            return True, mensagem
            
        except Exception as e:
            return False, f"Erro ao treinar modelo: {str(e)}"
    
    
    def predict(self, athleteData: Dict | pd.DataFrame) -> Tuple[bool, str | Dict]:
        """
        Prediz o cluster de um ou mais atletas.
        
        Args:
            athleteData: Dicionário ou DataFrame com dados do(s) atleta(s)
            
        Returns:
            Tupla (sucesso, resultado)
            resultado pode ser string (nome do cluster) ou dict com detalhes
        """
        if not self.isTrained:
            return False, "Modelo não foi treinado ainda"
        
        try:
            # Preparar features
            if isinstance(athleteData, dict):
                athleteData = pd.DataFrame([athleteData])
            
            X = self.prepareFeatures(athleteData)
            XScaled = self.scaler.transform(X)
            
            # Fazer predição
            predictions = self.knn.predict(XScaled)
            probabilities = self.knn.predict_proba(XScaled)
            
            # Se for apenas um atleta, retornar resultado simples
            if len(predictions) == 1:
                clusterCode = predictions[0]
                clusterName = self.clusterMapping[clusterCode]
                confidence = probabilities[0][clusterCode] * 100
                
                # Obter K vizinhos mais próximos
                distances, indices = self.knn.kneighbors(XScaled, n_neighbors=self.nNeighbors)
                
                resultado = {
                    'cluster': clusterName,
                    'confianca': f"{confidence:.1f}%",
                    'probabilidades': {
                        self.clusterMapping[i]: f"{prob*100:.1f}%"
                        for i, prob in enumerate(probabilities[0])
                    },
                    'distanciaMedia': float(distances[0].mean())
                }
                
                return True, resultado
            
            # Se forem múltiplos atletas, retornar lista
            resultados = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                clusterName = self.clusterMapping[pred]
                confidence = probs[pred] * 100
                
                resultados.append({
                    'indice': i,
                    'cluster': clusterName,
                    'confianca': f"{confidence:.1f}%"
                })
            
            return True, resultados
            
        except Exception as e:
            return False, f"Erro ao fazer predição: {str(e)}"
    
    
    def getModelMetrics(self, testData: pd.DataFrame | List[Dict] = None) -> Dict:
        """
        Retorna métricas detalhadas do modelo.
        
        Args:
            testData: Dados de teste (opcional)
            
        Returns:
            Dicionário com métricas do modelo
        """
        if not self.isTrained:
            return {'erro': 'Modelo não treinado'}
        
        metrics = {
            'nNeighbors': self.nNeighbors,
            'featureNames': self.featureNames,
            'nClusters': len(self.clusterMapping)
        }
        
        if testData is not None:
            try:
                X = self.prepareFeatures(testData)
                y = self.prepareLabels(testData)
                XScaled = self.scaler.transform(X)
                
                yPred = self.knn.predict(XScaled)
                
                metrics['acuracia'] = float((yPred == y).mean())
                metrics['silhouette'] = float(silhouette_score(XScaled, yPred))
                metrics['daviesBouldin'] = float(davies_bouldin_score(XScaled, yPred))
                
                # Relatório de classificação
                report = classification_report(
                    y, yPred,
                    target_names=list(self.clusterMapping.values()),
                    output_dict=True
                )
                metrics['reportClassificacao'] = report
                
            except Exception as e:
                metrics['erroMetricas'] = str(e)
        
        return metrics
    
    
    def findSimilarAthletes(self, athleteData: Dict, nSimilar: int = 5) -> Tuple[bool, List[Dict]]:
        """
        Encontra atletas similares baseado nas características.
        
        Args:
            athleteData: Dados do atleta de referência
            nSimilar: Número de atletas similares a retornar
            
        Returns:
            Tupla (sucesso, lista de atletas similares)
        """
        if not self.isTrained:
            return False, "Modelo não treinado"
        
        try:
            # Preparar features
            athleteDf = pd.DataFrame([athleteData])
            X = self.prepareFeatures(athleteDf)
            XScaled = self.scaler.transform(X)
            
            # Encontrar vizinhos mais próximos
            distances, indices = self.knn.kneighbors(XScaled, n_neighbors=nSimilar)
            
            similares = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                similares.append({
                    'posicao': i + 1,
                    'indice': int(idx),
                    'distancia': float(dist),
                    'similaridade': float(1 / (1 + dist))  # Converter distância em similaridade
                })
            
            return True, similares
            
        except Exception as e:
            return False, f"Erro ao buscar similares: {str(e)}"
    
    
    def optimizeKValue(self, trainData: pd.DataFrame, kRange: range = range(3, 21, 2)) -> Tuple[int, Dict]:
        """
        Encontra o melhor valor de K usando validação cruzada.
        
        Args:
            trainData: Dados de treinamento
            kRange: Range de valores de K para testar
            
        Returns:
            Tupla (melhor K, dicionário com resultados)
        """
        X = self.prepareFeatures(trainData)
        y = self.prepareLabels(trainData)
        XScaled = self.scaler.fit_transform(X)
        
        results = {}
        bestScore = 0
        bestK = self.nNeighbors
        
        for k in kRange:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            scores = cross_val_score(knn, XScaled, y, cv=5)
            meanScore = scores.mean()
            
            results[k] = {
                'acuraciaMedia': meanScore,
                'desvioPatrao': scores.std()
            }
            
            if meanScore > bestScore:
                bestScore = meanScore
                bestK = k
        
        # Atualizar modelo com melhor K
        self.nNeighbors = bestK
        self.knn = KNeighborsClassifier(n_neighbors=bestK, weights='distance')
        
        return bestK, results
    
    
    def saveModel(self, modelPath: str = 'kmeans_model.pkl') -> Tuple[bool, str]:
        """
        Salva o modelo treinado em disco.
        
        Args:
            modelPath: Caminho para salvar o modelo
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        if not self.isTrained:
            return False, "Modelo não foi treinado ainda"
        
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(modelPath) if os.path.dirname(modelPath) else '.', exist_ok=True)
            
            # Salvar modelo completo
            modelData = {
                'knn': self.knn,
                'scaler': self.scaler,
                'clusterMapping': self.clusterMapping,
                'featureNames': self.featureNames,
                'nNeighbors': self.nNeighbors
            }
            
            joblib.dump(modelData, modelPath)
            
            # Salvar também o scaler separadamente (compatibilidade)
            scalerPath = modelPath.replace('.pkl', '_scaler.pkl')
            joblib.dump(self.scaler, scalerPath)
            
            return True, f"Modelo salvo em {modelPath}"
            
        except Exception as e:
            return False, f"Erro ao salvar modelo: {str(e)}"
    
    
    def loadModel(self, modelPath: str = 'kmeans_model.pkl') -> Tuple[bool, str]:
        """
        Carrega um modelo previamente salvo.
        
        Args:
            modelPath: Caminho do modelo salvo
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            if not os.path.exists(modelPath):
                return False, f"Arquivo {modelPath} não encontrado"
            
            # Carregar modelo completo
            modelData = joblib.load(modelPath)
            
            self.knn = modelData['knn']
            self.scaler = modelData['scaler']
            self.clusterMapping = modelData['clusterMapping']
            self.featureNames = modelData['featureNames']
            self.nNeighbors = modelData['nNeighbors']
            self.reverseMapping = {v: k for k, v in self.clusterMapping.items()}
            self.isTrained = True
            
            return True, f"Modelo carregado de {modelPath}"
            
        except Exception as e:
            return False, f"Erro ao carregar modelo: {str(e)}"
    
    
    def getFeatureImportance(self, trainData: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula a importância relativa de cada feature.
        Usa correlação com os clusters como proxy.
        
        Args:
            trainData: Dados de treinamento
            
        Returns:
            Dicionário com importância de cada feature
        """
        try:
            X = self.prepareFeatures(trainData)
            y = self.prepareLabels(trainData)
            
            # Calcular correlação de cada feature com o target
            importance = {}
            for i, featureName in enumerate(self.featureNames):
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                importance[featureName] = abs(correlation)  # Valor absoluto da correlação
            
            # Normalizar para somar 1
            total = sum(importance.values())
            importance = {k: v/total for k, v in importance.items()}
            
            # Ordenar por importância
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance
            
        except Exception as e:
            return {'erro': str(e)}