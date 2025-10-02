from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

class analisadorAptidaoFisicaKnn:
    def __init__(self):
        self.scaler = StandardScaler()
        self.modeloKnnGeral = None
        self.modeloKnnCardio = None
        self.modeloKnnForca = None
        self.modeloKnnFlexibilidade = None
        self.modeloKnnAgilidade = None
        self.kVizinhos = 5
        
        # Criar dataset de treino simulado (em produção, usar dados reais)
        self.criarDatasetTreino()
        self.treinarModelos()

    def criarDatasetTreino(self):
        """Cria um dataset de treino simulado com dados de aptidão física"""
        np.random.seed(42)
        
        # Simular dados de 1000 pessoas com diferentes perfis
        nAmostras = 1000
        
        # Características base para diferentes perfis
        perfis = {
            'sedentario': {
                'idade': (25, 65), 'peso': (70, 120), 'altura': (1.6, 1.8),
                'distancia6min': (300, 500), 'abdominais1min': (5, 25),
                'testeSentarAlcancar': (-10, 5), 'arremessoMedicineBall': (2, 4),
                'saltoHorizontal': (80, 140), 'tempoQuadrado4x4': (15, 25),
                'tempoCorrida20m': (5, 8)
            },
            'moderado': {
                'idade': (20, 60), 'peso': (55, 90), 'altura': (1.6, 1.8),
                'distancia6min': (450, 650), 'abdominais1min': (20, 40),
                'testeSentarAlcancar': (0, 15), 'arremessoMedicineBall': (3, 6),
                'saltoHorizontal': (120, 180), 'tempoQuadrado4x4': (10, 18),
                'tempoCorrida20m': (4, 6)
            },
            'ativo': {
                'idade': (18, 55), 'peso': (55, 85), 'altura': (1.6, 1.8),
                'distancia6min': (600, 800), 'abdominais1min': (35, 55),
                'testeSentarAlcancar': (10, 25), 'arremessoMedicineBall': (5, 8),
                'saltoHorizontal': (160, 220), 'tempoQuadrado4x4': (8, 14),
                'tempoCorrida20m': (3.5, 5)
            },
            'atletico': {
                'idade': (18, 45), 'peso': (60, 80), 'altura': (1.65, 1.85),
                'distancia6min': (750, 1000), 'abdominais1min': (50, 80),
                'testeSentarAlcancar': (20, 35), 'arremessoMedicineBall': (7, 12),
                'saltoHorizontal': (200, 280), 'tempoQuadrado4x4': (6, 10),
                'tempoCorrida20m': (2.5, 4)
            }
        }
        
        dadosTreino = []
        
        for perfil, limites in perfis.items():
            nPorPerfil = nAmostras // 4
            
            for _ in range(nPorPerfil):
                pessoa = {}
                for caracteristica, (minVal, maxVal) in limites.items():
                    if caracteristica in ['idade', 'abdominais1min', 'saltoHorizontal']:
                        pessoa[caracteristica] = np.random.randint(minVal, maxVal)
                    else:
                        pessoa[caracteristica] = np.random.uniform(minVal, maxVal)
                
                # Calcular características derivadas
                imc = pessoa['peso'] / (pessoa['altura'] ** 2)
                pessoa['imc'] = imc
                pessoa['perimetroCintura'] = np.random.uniform(70, 110)
                pessoa['rce'] = pessoa['perimetroCintura'] / (pessoa['altura'] * 100)
                pessoa['sexo'] = np.random.choice(['masculino', 'feminino'])
                
                # Classificações específicas
                if perfil == 'sedentario':
                    pessoa['classificacaoGeral'] = 'Sedentário'
                    pessoa['aptidaoCardio'] = 'Fraco'
                    pessoa['aptidaoForca'] = 'Fraco'
                    pessoa['flexibilidade'] = 'Fraco'
                    pessoa['agilidade'] = 'Fraco'
                elif perfil == 'moderado':
                    pessoa['classificacaoGeral'] = 'Moderadamente Ativo'
                    pessoa['aptidaoCardio'] = 'Regular'
                    pessoa['aptidaoForca'] = 'Regular'
                    pessoa['flexibilidade'] = 'Regular'
                    pessoa['agilidade'] = 'Regular'
                elif perfil == 'ativo':
                    pessoa['classificacaoGeral'] = 'Ativo'
                    pessoa['aptidaoCardio'] = 'Bom'
                    pessoa['aptidaoForca'] = 'Bom'
                    pessoa['flexibilidade'] = 'Bom'
                    pessoa['agilidade'] = 'Bom'
                else:  # atletico
                    pessoa['classificacaoGeral'] = 'Atlético'
                    pessoa['aptidaoCardio'] = 'Excelente'
                    pessoa['aptidaoForca'] = 'Excelente'
                    pessoa['flexibilidade'] = 'Excelente'
                    pessoa['agilidade'] = 'Excelente'
                
                dadosTreino.append(pessoa)
        
        self.datasetTreino = pd.DataFrame(dadosTreino)

    def treinarModelos(self):
        """Treina os modelos KNN para diferentes aspectos da aptidão"""
        
        # Características para treino
        caracteristicasBase = [
            'idade', 'peso', 'altura', 'imc', 'rce', 
            'distancia6min', 'abdominais1min', 'testeSentarAlcancar',
            'arremessoMedicineBall', 'saltoHorizontal', 
            'tempoQuadrado4x4', 'tempoCorrida20m'
        ]
        
        # Preparar dados
        X = self.datasetTreino[caracteristicasBase].values
        
        # Normalizar características
        X_normalizado = self.scaler.fit_transform(X)
        
        # Treinar modelo geral
        yGeral = self.datasetTreino['classificacaoGeral'].values
        self.modeloKnnGeral = KNeighborsClassifier(n_neighbors=self.kVizinhos)
        self.modeloKnnGeral.fit(X_normalizado, yGeral)
        
        # Treinar modelos específicos
        yCardio = self.datasetTreino['aptidaoCardio'].values
        self.modeloKnnCardio = KNeighborsClassifier(n_neighbors=self.kVizinhos)
        self.modeloKnnCardio.fit(X_normalizado, yCardio)
        
        yForca = self.datasetTreino['aptidaoForca'].values
        self.modeloKnnForca = KNeighborsClassifier(n_neighbors=self.kVizinhos)
        self.modeloKnnForca.fit(X_normalizado, yForca)
        
        yFlexibilidade = self.datasetTreino['flexibilidade'].values
        self.modeloKnnFlexibilidade = KNeighborsClassifier(n_neighbors=self.kVizinhos)
        self.modeloKnnFlexibilidade.fit(X_normalizado, yFlexibilidade)
        
        yAgilidade = self.datasetTreino['agilidade'].values
        self.modeloKnnAgilidade = KNeighborsClassifier(n_neighbors=self.kVizinhos)
        self.modeloKnnAgilidade.fit(X_normalizado, yAgilidade)

    def calcularImc(self, peso, altura):
        """Calcula o Índice de Massa Corporal"""
        return peso / (altura ** 2)

    def calcularRce(self, circunferenciaCintura, altura):
        """Calcula a Razão Cintura-Estatura"""
        return circunferenciaCintura / (altura * 100)

    def classificarImc(self, imc):
        """Classifica o IMC"""
        if imc < 18.5:
            return "Abaixo do Peso"
        elif imc < 25:
            return "Peso Normal"
        elif imc < 30:
            return "Sobrepeso"
        elif imc < 35:
            return "Obesidade Grau 1"
        elif imc < 40:
            return "Obesidade Grau 2"
        else:
            return "Obesidade Grau 3"

    def classificarRce(self, rce):
        """Classifica a RCE"""
        if rce < 0.5:
            return "Baixo Risco"
        elif rce < 0.6:
            return "Risco Moderado"
        else:
            return "Alto Risco"

    def prepararDadosPredricao(self, dadosUsuario):
        """Prepara dados do usuário para predição"""
        caracteristicas = [
            dadosUsuario['idade'],
            dadosUsuario['peso'],
            dadosUsuario['altura'],
            dadosUsuario['imc'],
            dadosUsuario['rce'],
            dadosUsuario['distancia6min'],
            dadosUsuario['abdominais1min'],
            dadosUsuario['testeSentarAlcancar'],
            dadosUsuario['arremessoMedicineBall'],
            dadosUsuario['saltoHorizontal'],
            dadosUsuario['tempoQuadrado4x4'],
            dadosUsuario['tempoCorrida20m']
        ]
        
        return np.array(caracteristicas).reshape(1, -1)

    def obterProbabilidades(self, modelo, dadosNormalizados):
        """Obtém probabilidades de cada classe"""
        probabilidades = modelo.predict_proba(dadosNormalizados)[0]
        classes = modelo.classes_
        
        probDict = {}
        for i, classe in enumerate(classes):
            probDict[classe] = round(probabilidades[i] * 100, 1)
        
        return probDict

    def analisarResultados(self, dadosUsuario):
        """Analisa todos os resultados usando KNN"""
        resultados = {}
        
        # Cálculos básicos
        imc = self.calcularImc(dadosUsuario['peso'], dadosUsuario['altura'])
        rce = self.calcularRce(dadosUsuario['perimetroCintura'], dadosUsuario['altura'])
        
        # Adicionar cálculos aos dados do usuário
        dadosUsuario['imc'] = imc
        dadosUsuario['rce'] = rce
        
        resultados['imc'] = {
            'valor': round(imc, 2),
            'classificacao': self.classificarImc(imc)
        }
        
        resultados['rce'] = {
            'valor': round(rce, 3),
            'classificacao': self.classificarRce(rce)
        }
        
        # Preparar dados para KNN
        dadosParaPredricao = self.prepararDadosPredricao(dadosUsuario)
        dadosNormalizados = self.scaler.transform(dadosParaPredricao)
        
        # Predições KNN
        classificacaoGeral = self.modeloKnnGeral.predict(dadosNormalizados)[0]
        aptidaoCardio = self.modeloKnnCardio.predict(dadosNormalizados)[0]
        aptidaoForca = self.modeloKnnForca.predict(dadosNormalizados)[0]
        flexibilidade = self.modeloKnnFlexibilidade.predict(dadosNormalizados)[0]
        agilidade = self.modeloKnnAgilidade.predict(dadosNormalizados)[0]
        
        # Obter probabilidades
        probGeral = self.obterProbabilidades(self.modeloKnnGeral, dadosNormalizados)
        probCardio = self.obterProbabilidades(self.modeloKnnCardio, dadosNormalizados)
        probForca = self.obterProbabilidades(self.modeloKnnForca, dadosNormalizados)
        probFlexibilidade = self.obterProbabilidades(self.modeloKnnFlexibilidade, dadosNormalizados)
        probAgilidade = self.obterProbabilidades(self.modeloKnnAgilidade, dadosNormalizados)
        
        # Resultados das classificações
        resultados['classificacaoGeral'] = {
            'predicao': classificacaoGeral,
            'probabilidades': probGeral,
            'confianca': round(max(probGeral.values()), 1)
        }
        
        resultados['aptidaoCardiorrespiratoria'] = {
            'predicao': aptidaoCardio,
            'probabilidades': probCardio,
            'confianca': round(max(probCardio.values()), 1)
        }
        
        resultados['aptidaoForca'] = {
            'predicao': aptidaoForca,
            'probabilidades': probForca,
            'confianca': round(max(probForca.values()), 1)
        }
        
        resultados['flexibilidade'] = {
            'predicao': flexibilidade,
            'probabilidades': probFlexibilidade,
            'confianca': round(max(probFlexibilidade.values()), 1)
        }
        
        resultados['agilidade'] = {
            'predicao': agilidade,
            'probabilidades': probAgilidade,
            'confianca': round(max(probAgilidade.values()), 1)
        }
        
        # Métricas adicionais
        resultados['metricas'] = {
            'kVizinhos': self.kVizinhos,
            'tamanhoDataset': len(self.datasetTreino),
            'algoritmo': 'K-Nearest Neighbors'
        }
        
        return resultados

    def ajustarKVizinhos(self, novoK):
        """Permite ajustar o número de vizinhos e retreinar"""
        self.kVizinhos = novoK
        self.treinarModelos()

# Instância do analisador
analisador = analisadorAptidaoFisicaKnn()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analisar', methods=['POST'])
def analisar():
    try:
        dadosUsuario = {
            'idade': int(request.form['idade']),
            'sexo': request.form['sexo'],
            'peso': float(request.form['peso']),
            'altura': float(request.form['altura']),
            'envergadura': float(request.form['envergadura']),
            'perimetroCintura': float(request.form['perimetroCintura']),
            'distancia6min': float(request.form['distancia6min']),
            'testeSentarAlcancar': float(request.form['testeSentarAlcancar']),
            'abdominais1min': int(request.form['abdominais1min']),
            'arremessoMedicineBall': float(request.form['arremessoMedicineBall']),
            'saltoHorizontal': float(request.form['saltoHorizontal']),
            'tempoQuadrado4x4': float(request.form['tempoQuadrado4x4']),
            'tempoCorrida20m': float(request.form['tempoCorrida20m'])
        }
        
        resultados = analisador.analisarResultados(dadosUsuario)
        
        return jsonify({
            'sucesso': True,
            'resultados': resultados
        })
        
    except Exception as e:
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 400

@app.route('/ajustar-k', methods=['POST'])
def ajustarK():
    try:
        novoK = int(request.form['k'])
        if novoK < 1 or novoK > 20:
            raise ValueError("K deve estar entre 1 e 20")
        
        analisador.ajustarKVizinhos(novoK)
        
        return jsonify({
            'sucesso': True,
            'mensagem': f'Número de vizinhos ajustado para {novoK}'
        })
        
    except Exception as e:
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 400

@app.route('/dataset-info')
def datasetInfo():
    """Retorna informações sobre o dataset de treino"""
    try:
        info = {
            'tamanho': len(analisador.datasetTreino),
            'colunas': list(analisador.datasetTreino.columns),
            'distribuicao': analisador.datasetTreino['classificacaoGeral'].value_counts().to_dict(),
            'kAtual': analisador.kVizinhos
        }
        
        return jsonify({
            'sucesso': True,
            'info': info
        })
        
    except Exception as e:
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)