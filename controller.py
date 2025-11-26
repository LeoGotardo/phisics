import locale, sys, uuid, pandas as pd, zipfile, datetime, matplotlib.pyplot as plt, io, numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples, classification_report
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, PageBreak, Paragraph, Spacer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from flask_sqlalchemy import SQLAlchemy
from sklearn.decomposition import PCA
from reportlab.lib import colors
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from flask_login import UserMixin
from sqlalchemy import func
from typing import Literal
from flask import Flask


class Config:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    SECRET_KEY = "secret_key"
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SECRET_KEY'] = SECRET_KEY
    db = SQLAlchemy(app)
    session = db.session
    
    
class Athlete(UserMixin, Config.db.Model):
    __tablename__ = 'athletes'
    id = Config.db.Column(Config.db.String(32), primary_key=True, default=lambda: str(uuid.uuid4()))
    nome = Config.db.Column(Config.db.String(50), nullable=False)
    data_nascimento = Config.db.Column(Config.db.Date, nullable=False)
    sexo = Config.db.Column(Config.db.Integer, nullable=False)  # 0=M, 1=F
    estatura = Config.db.Column(Config.db.Float, nullable=False)
    envergadura = Config.db.Column(Config.db.Float, nullable=False)
    arremesso = Config.db.Column(Config.db.Float, nullable=False)
    salto_horizontal = Config.db.Column(Config.db.Float, nullable=False)
    abdominais = Config.db.Column(Config.db.Float, nullable=False)
    cluster = Config.db.Column(Config.db.String(20), nullable=True)
    
    
    def __init__(self, nome, data_nascimento, sexo, estatura, envergadura, arremesso, salto_horizontal, abdominais, cluster=None):
        self.nome = nome
        self.data_nascimento = data_nascimento
        self.sexo = sexo
        self.estatura = estatura
        self.envergadura = envergadura
        self.arremesso = arremesso
        self.salto_horizontal = salto_horizontal
        self.abdominais = abdominais
        self.cluster = cluster
        
    def dict(self):
        return {
            'id': self.id,
            'nome': self.nome,
            'data_nascimento': self.data_nascimento,
            'sexo': 'M' if self.sexo == 0 else 'F',
            'estatura': self.estatura,
            'envergadura': self.envergadura,
            'arremesso': self.arremesso,
            'salto_horizontal': self.salto_horizontal,
            'abdominais': self.abdominais,
            'cluster': self.cluster
        }
        
        
class Model:
    def __init__(self):
        self.CLUSTERS = ['Elite', 'Competitivo', 'Intermediário', 'Iniciante']
        self.COLUMNS = ['nome', 'data_nascimento', 'sexo', 'estatura', 'envergadura', 'arremesso', 'salto_horizontal', 'abdominais']
        self.FEATURE_COLUMNS = ['estatura', 'envergadura', 'arremesso', 'salto_horizontal', 'abdominais']
        
        self.db = Config.db
        self.session = Config.session
        
        # Modelos
        self.kmeans = None
        self.knn_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Dados treinados
        self.data = None
        self.data_scaled = None
        self.labels = None
        
        self.create_tables()
        
    
    def create_tables(self):
        with Config.app.app_context():
            self.db.create_all()
    
    
    def loadCSVData(self, file):
        """Carrega dados de um arquivo CSV"""
        try:
            data = pd.read_csv(file)
            
            if not self.vrfyCSVColumns(data):
                raise ValueError("Erro ao carregar dados. Verifique se os nomes dos atributos estão corretos.")
            
            data = self.sanitizeCSV(data) 
            data_dict = data.to_dict(orient='records')
            
            success = self.createAthletes(data_dict)
            if success:
                return True, f'{len(data_dict)} atletas carregados com sucesso'
            return False, 'Erro ao criar atletas'
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
        

    def vrfyCSVColumns(self, data: pd.DataFrame):
        """Verifica se as colunas do CSV estão corretas"""
        try:
            data.columns = data.columns.str.lower().str.strip()
            for column in self.COLUMNS:
                if column not in data.columns:
                    return False
            return True
        except (KeyError, AttributeError):
            return False


    def sanitizeCSV(self, data: pd.DataFrame):
        """Sanitiza e valida os dados do CSV"""
        def isNum(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return False
        
        def isDate(x):
            try:
                return datetime.datetime.strptime(str(x), '%Y-%m-%d')
            except (ValueError, TypeError):
                return False
        
        def isSex(x):
            val = str(x).upper().strip()
            if val == 'M':
                return 0
            if val == 'F':
                return 1
            return False
        
        data = data.copy()
        
        # Validar colunas numéricas
        for column in self.FEATURE_COLUMNS:
            converted = data[column].apply(isNum)
            if (converted == False).any():
                raise ValueError(f"Coluna '{column}' contém valores não numéricos")
            data[column] = converted
        
        # Validar datas
        converted_dates = data['data_nascimento'].apply(isDate)
        if (converted_dates == False).any():
            raise ValueError("Coluna 'data_nascimento' contém datas inválidas")
        data['data_nascimento'] = converted_dates
        
        # Validar sexo
        converted_sex = data['sexo'].apply(isSex)
        if (converted_sex == False).any():
            raise ValueError("Coluna 'sexo' contém valores inválidos (use 'M' ou 'F')")
        data['sexo'] = converted_sex
        
        return data
    
    
    def trainKMeansModel(self, n_clusters=4):
        """Treina o modelo KMeans para criar clusters iniciais"""
        try:
            athletes = self.session.query(Athlete).all()
            
            if len(athletes) < n_clusters:
                return False, f'Necessário pelo menos {n_clusters} atletas para treinar o modelo'
            
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
            
            # Padronizar dados
            self.data_scaled = self.scaler.fit_transform(self.data)
            
            # Treinar KMeans
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels = self.kmeans.fit_predict(self.data_scaled)
            
            # Calcular média de performance por cluster para ordenação
            cluster_means = []
            for i in range(n_clusters):
                cluster_data = self.data[self.labels == i]
                cluster_means.append(np.mean(cluster_data))
            
            # Ordenar clusters do melhor para o pior
            cluster_order = np.argsort(cluster_means)[::-1]
            cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
            
            # Atualizar labels com a nova ordem
            self.labels = np.array([cluster_mapping[label] for label in self.labels])
            
            # Atualizar clusters dos atletas
            for i, athlete in enumerate(athletes):
                cluster_idx = self.labels[i]
                athlete.cluster = self.CLUSTERS[cluster_idx]
            
            self.session.commit()
            
            return True, 'Modelo KMeans treinado com sucesso'
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def trainKNNModel(self, n_neighbors=5):
        """Treina o modelo KNN com base nos clusters criados pelo KMeans"""
        try:
            # Primeiro treinar KMeans se necessário
            if self.kmeans is None or self.labels is None:
                status, msg = self.trainKMeansModel()
                if status != True:
                    return status, msg
            
            # Preparar dados para KNN
            X_train = self.data_scaled
            y_train = self.labels
            
            # Codificar labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            
            # Treinar KNN
            self.knn_model = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(X_train)))
            self.knn_model.fit(X_train, y_train_encoded)
            
            return True, 'Modelo KNN treinado com sucesso'
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def predictCluster(self, athlete_data: dict):
        """Prediz o cluster de um novo atleta usando KNN"""
        try:
            if self.knn_model is None:
                return False, 'Modelo não treinado. Execute trainKNNModel() primeiro.'
            
            # Preparar features
            features = np.array([[
                athlete_data['estatura'],
                athlete_data['envergadura'],
                athlete_data['arremesso'],
                athlete_data['salto_horizontal'],
                athlete_data['abdominais']
            ]])
            
            # Padronizar
            features_scaled = self.scaler.transform(features)
            
            # Predizer
            cluster_encoded = self.knn_model.predict(features_scaled)[0]
            cluster_idx = self.label_encoder.inverse_transform([cluster_encoded])[0]
            
            # Probabilidades
            probabilities = self.knn_model.predict_proba(features_scaled)[0]
            
            probs_dict = {}
            for i, prob in enumerate(probabilities):
                original_idx = self.label_encoder.inverse_transform([i])[0]
                cluster_name = self.CLUSTERS[original_idx]
                probs_dict[cluster_name] = float(prob * 100)
            
            cluster_name = self.CLUSTERS[cluster_idx]
            
            return True, {
                'cluster': cluster_name,
                'cluster_idx': int(cluster_idx),
                'probabilidades': probs_dict
            }
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def getDashboardInfo(self):
        """Retorna informações do dashboard com estatísticas dos clusters"""
        try:
            total_athletes = self.session.query(Athlete).count()
            
            cluster_info = {}
            
            for cluster_name in self.CLUSTERS:
                athletes_in_cluster = self.session.query(Athlete).filter(Athlete.cluster == cluster_name)
                count = athletes_in_cluster.count()
                
                if count > 0:
                    cluster_info[cluster_name] = {
                        'label': cluster_name,
                        'count': count,
                        'estatura_media': athletes_in_cluster.with_entities(func.avg(Athlete.estatura)).scalar(),
                        'envergadura_media': athletes_in_cluster.with_entities(func.avg(Athlete.envergadura)).scalar(),
                        'arremesso_medio': athletes_in_cluster.with_entities(func.avg(Athlete.arremesso)).scalar(),
                        'salto_medio': athletes_in_cluster.with_entities(func.avg(Athlete.salto_horizontal)).scalar(),
                        'abdominais_medio': athletes_in_cluster.with_entities(func.avg(Athlete.abdominais)).scalar()
                    }
                else:
                    cluster_info[cluster_name] = {
                        'label': cluster_name,
                        'count': 0,
                        'estatura_media': 0,
                        'envergadura_media': 0,
                        'arremesso_medio': 0,
                        'salto_medio': 0,
                        'abdominais_medio': 0
                    }
            
            return True, {
                'total_atletas': total_athletes,
                'clusters': cluster_info
            }
                
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def getAthletes(self, sort='nome', sort_order='asc', query=None, paginated=False, page=1, per_page=10):
        """Retorna lista de atletas com opções de filtro e paginação"""
        try:
            sort_options = {
                'nome': Athlete.nome,
                'data_nascimento': Athlete.data_nascimento,
                'sexo': Athlete.sexo,
                'estatura': Athlete.estatura,
                'envergadura': Athlete.envergadura,
                'arremesso': Athlete.arremesso,
                'salto_horizontal': Athlete.salto_horizontal,
                'abdominais': Athlete.abdominais,
                'cluster': Athlete.cluster
            }
            
            sort_column = sort_options.get(sort, Athlete.nome)
            
            with Config.app.app_context():
                base_query = self.session.query(Athlete)
            
            if query:
                base_query = base_query.filter(Athlete.nome.like(f'%{query}%'))
                
            if sort_order == 'desc':
                base_query = base_query.order_by(sort_column.desc())
            else:
                base_query = base_query.order_by(sort_column.asc())
                
            if paginated:
                paginated_results = base_query.paginate(page=page, per_page=per_page, error_out=False)
                
                athletes = [athlete.dict() for athlete in paginated_results.items]
                
                return True, {
                    'items': athletes,
                    'pagination': {
                        'current_page': paginated_results.page,
                        'total_pages': paginated_results.pages,
                        'total': paginated_results.total,
                        'per_page': paginated_results.per_page,
                        'has_prev': paginated_results.has_prev,
                        'has_next': paginated_results.has_next
                    }
                }
            else:
                athletes = base_query.all()
                return True, [athlete.dict() for athlete in athletes]
                
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def createAthletes(self, data):
        """Cria um ou mais atletas no banco de dados"""
        try:
            with Config.app.app_context():
                if isinstance(data, list):
                    for item in data:
                        athlete = Athlete(
                            nome=item['nome'],
                            data_nascimento=item['data_nascimento'],
                            sexo=item['sexo'],
                            estatura=item['estatura'],
                            envergadura=item['envergadura'],
                            arremesso=item['arremesso'],
                            salto_horizontal=item['salto_horizontal'],
                            abdominais=item['abdominais']
                        )
                        self.session.add(athlete)
                    
                elif isinstance(data, dict):
                    athlete = Athlete(
                        nome=data['nome'],
                        data_nascimento=data['data_nascimento'],
                        sexo=data['sexo'],
                        estatura=data['estatura'],
                        envergadura=data['envergadura'],
                        arremesso=data['arremesso'],
                        salto_horizontal=data['salto_horizontal'],
                        abdominais=data['abdominais']
                    )
                    self.session.add(athlete)
                
                self.session.commit()
                return True
                
        except Exception as e:
            self.session.rollback()
            raise ValueError(f'Erro ao criar atleta: {e}')
    
    
    def getModelHealth(self):
        """Retorna métricas de saúde do modelo"""
        try:
            if self.kmeans is None or self.data_scaled is None:
                return False, 'Modelo não treinado'
            
            labels = self.labels
            X = self.data_scaled
            
            # Silhouette Score
            silhouette_avg = silhouette_score(X, labels)
            silhouette_samples_values = silhouette_samples(X, labels)
            
            silhouette_by_cluster = {}
            for i in range(len(self.CLUSTERS)):
                cluster_silhouette = silhouette_samples_values[labels == i]
                if len(cluster_silhouette) > 0:
                    silhouette_by_cluster[self.CLUSTERS[i]] = {
                        'mean': float(np.mean(cluster_silhouette)),
                        'size': int(np.sum(labels == i))
                    }
            
            # Davies-Bouldin Index
            davies_bouldin = davies_bouldin_score(X, labels)
            
            # Inércia
            inertia = self.kmeans.inertia_
            
            # PCA para visualização
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            pca_data = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'clusters': {}
            }
            
            for i in range(len(self.CLUSTERS)):
                cluster_mask = labels == i
                if np.sum(cluster_mask) > 0:
                    pca_data['clusters'][self.CLUSTERS[i]] = {
                        'x': X_pca[cluster_mask, 0].tolist(),
                        'y': X_pca[cluster_mask, 1].tolist(),
                        'size': int(np.sum(cluster_mask))
                    }
            
            health_metrics = {
                'silhouette': {
                    'score': float(silhouette_avg),
                    'by_cluster': silhouette_by_cluster,
                    'interpretation': 'good' if silhouette_avg > 0.5 else 'moderate' if silhouette_avg > 0.25 else 'poor'
                },
                'davies_bouldin': {
                    'score': float(davies_bouldin),
                    'interpretation': 'good' if davies_bouldin < 1.0 else 'moderate' if davies_bouldin < 2.0 else 'poor'
                },
                'inertia': float(inertia),
                'pca': pca_data,
                'n_clusters': len(self.CLUSTERS),
                'n_samples': int(len(X))
            }
            
            return True, health_metrics
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def getDBStats(self):
        """Retorna estatísticas do banco de dados"""
        try:
            total = self.session.query(Athlete).count()
            male = self.session.query(Athlete).filter(Athlete.sexo == 0).count()
            female = self.session.query(Athlete).filter(Athlete.sexo == 1).count()
            
            stats = {
                'total_atletas': total,
                'masculino': male,
                'feminino': female,
                'por_cluster': {}
            }
            
            for cluster in self.CLUSTERS:
                count = self.session.query(Athlete).filter(Athlete.cluster == cluster).count()
                stats['por_cluster'][cluster] = count
            
            return stats
            
        except Exception as e:
            return {}
    
    
    def generateCSV(self, data):
        """Gera CSV a partir dos dados"""
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        return csv_data.encode('utf-8')
    
    
    def exportData(self, full_data=False):
        """Exporta dados em CSV ou PDF completo"""
        try:
            status, data = self.getAthletes(paginated=False)
            
            if not status or len(data) == 0:
                return False, 'Não há dados para exportar'
            
            if full_data:
                return self.generateFullReport(data)
            else:
                csv = self.generateCSV(data)
                return True, csv
                
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def generateFullReport(self, data):
        """Gera relatório completo em PDF"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()
            
            # Título
            title = Paragraph("Relatório de Atletas - Powerlifting", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Estatísticas
            db_stats = self.getDBStats()
            stats_text = f"""
            Total de Atletas: {db_stats['total_atletas']}<br/>
            Masculino: {db_stats['masculino']}<br/>
            Feminino: {db_stats['feminino']}<br/>
            """
            elements.append(Paragraph(stats_text, styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Tabela de atletas
            table_data = [['Nome', 'Sexo', 'Estatura', 'Cluster']]
            for athlete in data[:20]:  # Limitar para não ficar muito grande
                table_data.append([
                    athlete['nome'],
                    athlete['sexo'],
                    f"{athlete['estatura']:.2f}",
                    athlete['cluster'] or 'N/A'
                ])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
            
            doc.build(elements)
            buffer.seek(0)
            return True, buffer.getvalue()
            
        except Exception as e:
            return -1, f'{type(e).__name__}: {e} in line {sys.exc_info()[-1].tb_lineno}'
    
    
    def genGraphs(self):
        """Gera gráficos de visualização dos clusters"""
        try:
            if self.data_scaled is None or self.labels is None:
                return []
            
            graphs = []
            
            # Gráfico PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.data_scaled)
            
            plt.figure(figsize=(10, 6))
            for i, cluster in enumerate(self.CLUSTERS):
                mask = self.labels == i
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cluster, alpha=0.6)
            
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.title('Visualização PCA dos Clusters')
            plt.legend()
            plt.grid(True)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graphs.append(buf.getvalue())
            plt.close()
            
            return graphs
            
        except Exception as e:
            return []