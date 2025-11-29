class KNNModel:
    def setupKMeans(self, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters)
    
        
    def trainModel(self, nClusters=None):
        """Treina o modelo KMeans com os dados dos atletas"""
        try:
            athletes = self.session.query(Athlete).all()
            
            if len(athletes) == 0:
                return False, 'Não há atletas cadastrados para treinar o modelo'
            
            # Preparar dados
            data = []
            for athlete in athletes:
                data.append([
                    athlete.estatura,
                    athlete.envergadura,
                    athlete.arremesso,
                    athlete.salto_horizontal,
                    athlete.abdominais
                ])
            
            self.data = np.array(data)
            
            # Definir número de clusters
            if nClusters is None:
                nClusters = min(4, len(athletes))
            
            # Padronizar dados
            self.dataScaled = self.scaler.fit_transform(self.data)
            
            # Treinar KMeans
            self.kmeans = KMeans(n_clusters=nClusters, random_state=42)
            self.labels = self.kmeans.fit_predict(self.dataScaled)
            
            # Atualizar clusters dos atletas
            for i, athlete in enumerate(athletes):
                clusterIdx = self.labels[i]
                athlete.cluster = self.CLUSTERS[clusterIdx] if clusterIdx < len(self.CLUSTERS) else f'Cluster {clusterIdx}'
            
            self.session.commit()
            
            return True, 'Modelo treinado com sucesso'
            
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}')
    
    
    def predictCluster(self, athleteData: dict):
        """Prediz o cluster de um novo atleta"""
        try:
            if self.kmeans is None:
                return False, 'Modelo não treinado'
            
            # Preparar dados
            features = np.array([[
                athleteData['estatura'],
                athleteData['envergadura'],
                athleteData['arremesso'],
                athleteData['saltoHorizontal'],
                athleteData['abdominais']
            ]])
            
            # Padronizar e predizer
            featuresScaled = self.scaler.transform(features)
            clusterIdx = self.kmeans.predict(featuresScaled)[0]
            probabilities = self.getClusterProbabilities(featuresScaled)
            
            clusterName = self.CLUSTERS[clusterIdx] if clusterIdx < len(self.CLUSTERS) else f'Cluster {clusterIdx}'
            
            return True, {
                'cluster': clusterName,
                'clusterIdx': int(clusterIdx),
                'probabilities': probabilities
            }
            
        except Exception as e:
            return -1, str(f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}')
    
    
    def getClusterProbabilities(self, featuresScaled):
        """Calcula probabilidades de pertencer a cada cluster baseado na distância"""
        distances = self.kmeans.transform(featuresScaled)[0]
        invDistances = 1 / (distances + 1e-10)
        probabilities = invDistances / invDistances.sum()
        
        result = {}
        for i, prob in enumerate(probabilities):
            clusterName = self.CLUSTERS[i] if i < len(self.CLUSTERS) else f'Cluster {i}'
            result[clusterName] = float(prob * 100)
        
        return result