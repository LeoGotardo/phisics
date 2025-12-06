import traceback, sys, json

from flask import Flask, render_template, request, redirect, url_for, flash, Blueprint, send_file
from src.model.model import Model
from src.config import Config
from datetime import datetime
from icecream import ic
from io import BytesIO


class Controller:
    def __init__(self, app):
        self.app: Flask = app
        self.model = Model()
        self.blueprint = Blueprint('view', __name__, template_folder='templates')
        
        self.app.register_blueprint(self.blueprint)
        self.defineRouters()
        
        self.app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
        
        
    def defineRouters(self):
        self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
        self.app.add_url_rule('/analise', view_func=self.review, methods=['GET', 'POST'])
        self.app.add_url_rule('/cadastro', view_func=self.createAthlete, methods=['GET', 'POST'])
        self.app.add_url_rule('/view', view_func=self.viewPage, methods=['GET'])
        self.app.add_url_rule('/loadCSVData', view_func=self.loadCSVData, methods=['POST'])
        
        # CRUD de Atletas
        self.app.add_url_rule('/atleta/editar/<string:athlete_id>', view_func=self.editAthlete, methods=['GET', 'POST'])
        self.app.add_url_rule('/atleta/excluir/<string:athlete_id>', view_func=self.deleteAthleteRoute, methods=['DELETE'])
        
        # Aliases para melhor navegação
        self.app.add_url_rule('/dashboard', 'renderDashboard', self.index, methods=['GET'])
        self.app.add_url_rule('/cadastro', 'renderCadastro', self.createAthlete, methods=['GET'])
        self.app.add_url_rule('/analise', 'renderReview', self.review, methods=['GET'])
        self.app.add_url_rule('/visualizacao', 'renderViewPage', self.viewPage, methods=['GET'])
    
    
    def handleException(self, e):
        excType, excValue, excTraceback = sys.exc_info()

        if excTraceback:
            lastFrame = traceback.extract_tb(excTraceback)[-1]
            errorFile = lastFrame.filename
            errorLine = lastFrame.lineno
            errorFunction = lastFrame.name
            errorCode = lastFrame.line if lastFrame.line else "N/A"
        else:
            errorFile = "Desconhecido"
            errorLine = "N/A"
            errorFunction = "N/A"
            errorCode = "N/A"

        fullTraceback = ''.join(traceback.format_exception(excType, excValue, excTraceback))

        errorDetails = None
        debugInfo = None

        if self.app.config.get('DEBUG'):
            errorDetails = str(e)
            debugInfo = {
                'file': errorFile.split('/')[-1] if errorFile else 'N/A',
                'line': errorLine,
                'function': errorFunction,
                'code': errorCode,
                'fullPath': errorFile,
                'traceback': fullTraceback
            }

        if hasattr(e, 'code'):
            return render_template('errorPage.html',
                                errorCode=e.code,
                                errorMessage=getattr(e, 'description', 'Erro desconhecido'),
                                errorDetails=errorDetails,
                                debugInfo=debugInfo), e.code

        return render_template('errorPage.html',
                            errorCode=500,
                            erroMessage="Ocorreu um erro inesperado. Nossa equipe foi notificada.",
                            errorDetails=errorDetails,
                            debugInfo=debugInfo), 500
    
    
    def index(self):
        status, info = self.model.getDashboardInfo()
        if status == True:
            return render_template('dashboard.html', **info)
        if status == -1:
            raise Exception(info)
        
        
    def loadCSVData(self):
        """
        Importa dados de arquivo CSV enviado pelo usuário.
        Valida, sanitiza e insere os atletas no banco de dados.
        """
        if request.method == 'POST':
            try:
                # Verificar se arquivo foi enviado
                if 'csvFile' not in request.files:
                    flash('Nenhum arquivo foi enviado', category='error')
                    return redirect(url_for('renderCadastro'))
                
                file = request.files['csvFile']
                
                # Verificar se arquivo tem nome
                if file.filename == '':
                    flash('Arquivo sem nome', category='error')
                    return redirect(url_for('renderCadastro'))
                
                # Verificar extensão
                if not file.filename.lower().endswith('.csv'):
                    flash('Apenas arquivos CSV são permitidos', category='error')
                    return redirect(url_for('renderCadastro'))
                
                # Ler o arquivo em bytes
                file_bytes = file.read()
                
                # Processar CSV através do model
                status, response = self.model.loadCSVData(file_bytes)
                
                if status == -1:
                    raise Exception(response)
                    
                if status == True:
                    flash(f'✓ {response}', category='success')
                else:
                    flash(f'⚠ {response}', category='warning')
                
                return redirect(url_for('renderCadastro'))
                
            except Exception as e:
                error_msg = f'{type(e).__name__}: {e}'
                flash(f'Erro ao importar CSV: {error_msg}', category='error')
                return redirect(url_for('renderCadastro'))
        else:
            return render_template('404.html'), 404
        
    
    def createAthlete(self):
        match request.method:
            case 'POST':
                dataDict = {
                'nome' : request.form.get('nome'),
                'dataNascimento' : datetime.strptime(request.form.get('dataNascimento'), '%Y-%m-%d'),
                'sexo' : request.form.get('sexo'),
                'altura' : request.form.get('altura'),
                'envergadura' : request.form.get('envergadura'),
                'arremesso' : request.form.get('arremesso'),
                'saltoHorizontal' : request.form.get('saltoHorizontal'),
                'abdominais' : request.form.get('abdominais'),
                }
                
                status, response = self.model.putAthlete(dataDict)
                if status == -1:
                    raise Exception(response)
                if status == True:
                    flash(response, category='success')
                else:
                    flash(response, category='error')
                return render_template('cadastro.html')
            case "GET":
                return render_template('cadastro.html')
            case _:
                return render_template('404.html')
    
    
    def review(self):
        """
        Página de análise com filtros e ordenação.
        """
        match request.method:
            case 'GET':
                # Pegar parâmetros da URL
                sort = request.args.get('sort', 'nome')
                sortOrder = request.args.get('sortOrder', 'asc')
                query = request.args.get('query', '')
                cluster = request.args.get('cluster', 'all')
                ageRange = request.args.get('ageRange', 'all')
                page = request.args.get('page', '1')
                perPage = request.args.get('perPage', '10')
                
                # CORREÇÃO: Processar ageRange corretamente
                ageRangeProcessed = None
                if ageRange != 'all' and ageRange:
                    # Converter faixa etária em datas de nascimento
                    from datetime import datetime, timedelta
                    hoje = datetime.now()
                    
                    if ageRange == '10-18':
                        # Nascidos entre (hoje - 18 anos) e (hoje - 10 anos)
                        data_max = (hoje - timedelta(days=10*365)).strftime('%Y-%m-%d')
                        data_min = (hoje - timedelta(days=18*365)).strftime('%Y-%m-%d')
                        ageRangeProcessed = (data_min, data_max)
                        
                    elif ageRange == '19-29':
                        data_max = (hoje - timedelta(days=19*365)).strftime('%Y-%m-%d')
                        data_min = (hoje - timedelta(days=29*365)).strftime('%Y-%m-%d')
                        ageRangeProcessed = (data_min, data_max)
                        
                    elif ageRange == '30+':
                        data_max = (hoje - timedelta(days=30*365)).strftime('%Y-%m-%d')
                        data_min = '1900-01-01'  # Data muito antiga
                        ageRangeProcessed = (data_min, data_max)
                
                # Processar cluster
                if cluster == 'all':
                    cluster = None
                
                success, athletes = self.model.getAthletes(
                    sort=sort,
                    sortOrder=sortOrder,
                    query=query,
                    cluster=cluster,
                    ageRange=ageRangeProcessed,
                    page=page,
                    per_page=perPage,
                    paginated=True
                )
                
                if success == True:
                    athletes['filters']['ageRange'] = ageRange
                    return render_template('analise.html', athletes=athletes)
                else:
                    raise Exception(athletes)
            
            case _:
                return render_template('404.html'), 404


    def viewPage(self):
        """
        Renderiza a página de visualização com dados estatísticos.
        Gera gráficos PCA, correlação, core strength, radar e métricas de qualidade.
        """
        match request.method:
            case 'GET':
                success, info = self.model.getViewInfo()
                
                if success == True:
                    if not isinstance(info, dict):
                        flash('Erro: dados retornados em formato inválido', category='error')
                        info = self._getEmptyViewData()
                    
                    requiredKeys = ['pca_data', 'variance_explained', 'potencia_data', 
                                'core_data', 'perfil_data', 'metricas', 'estatisticas']
                    
                    missingKeys = [key for key in requiredKeys if key not in info]
                    
                    if missingKeys:
                        flash(f'Aviso: dados incompletos - {", ".join(missingKeys)}', category='warning')
                        for key in missingKeys:
                            info[key] = self._getEmptyData(key)
                    
                    return render_template('view.html', info=info)
                    
                elif success == False:
                    flash(info, category='warning')
                    emptyData = self._getEmptyViewData()
                    return render_template('view.html', info=emptyData)
                
                else:
                    raise Exception(info)
            
            case _:
                return render_template('404.html'), 404


    def _getEmptyViewData(self):
        """Retorna estrutura vazia para quando não há dados"""
        return {
            'pca_data': [],
            'variance_explained': {'pc1': 0, 'pc2': 0, 'total': 0},
            'potencia_data': {
                'x': [], 
                'y': [], 
                'correlacao': 0, 
                'p_value': '1.0'
            },
            'core_data': [],
            'perfil_data': [],
            'metricas': {
                'silhouette': 0, 
                'davies_bouldin': 0, 
                'inertia': 0
            },
            'estatisticas': {
                'total_atletas': 0, 
                'distribuicao': {}, 
                'features_analisadas': 0
            }
        }


    def _getEmptyData(self, key: str):
        """Retorna dados vazios para uma chave específica"""
        emptyDataMap = {
            'pca_data': [],
            'variance_explained': {'pc1': 0, 'pc2': 0, 'total': 0},
            'potencia_data': {'x': [], 'y': [], 'correlacao': 0, 'p_value': '1.0'},
            'core_data': [],
            'perfil_data': [],
            'metricas': {'silhouette': 0, 'davies_bouldin': 0, 'inertia': 0},
            'estatisticas': {'total_atletas': 0, 'distribuicao': {}, 'features_analisadas': 0}
        }
        
        return emptyDataMap.get(key, None)


    def editAthlete(self, athlete_id):
        """
        Página e ação de edição de atleta.
        """
        match request.method:
            case 'GET':
                success, athlete = self.model.getAthleteById(athlete_id)
                ic(athlete)
                
                if success == True:
                    athlete_data = athlete.dict()
                    athlete_data['dataNascimento'] = athlete.dataNascimento.strftime('%Y-%m-%d')
                    athlete_data['sexo'] = str(athlete_data['sexo']).replace('0', 'Masculino').replace('1', 'Feminino')
                    return render_template('editAthlete.html', athlete=athlete_data)
                elif success == False:
                    flash(athlete, category='error')
                    return redirect(url_for('renderReview'))
                else:
                    raise Exception(athlete)
            
            case 'POST':
                dataDict = {
                    'nome': request.form.get('nome'),
                    'dataNascimento': datetime.strptime(request.form.get('dataNascimento'), '%Y-%m-%d'),
                    'sexo': request.form.get('sexo'),
                    'altura': request.form.get('altura'),
                    'envergadura': request.form.get('envergadura'),
                    'arremesso': request.form.get('arremesso'),
                    'saltoHorizontal': request.form.get('saltoHorizontal'),
                    'abdominais': request.form.get('abdominais'),
                }
                
                success, message = self.model.updateAthlete(athlete_id, dataDict)
                
                if success == True:
                    flash(message, category='success')
                elif success == False:
                    flash(message, category='error')
                else:
                    raise Exception(message)
                
                return redirect(url_for('renderReview'))
            
            case _:
                return render_template('404.html'), 404


    def deleteAthleteRoute(self, athlete_id):
        """
        Rota para excluir atleta.
        """
        if request.method == 'DELETE':
            success, message = self.model.deleteAthlete(athlete_id)
            
            if success == True:
                return {'success': True, 'message': message}, 200
            elif success == False:
                return {'success': False, 'message': message}, 404
            else:
                return {'success': False, 'message': message}, 500
        else:
            return render_template('404.html'), 404
        
        
if __name__ == '__main__':
    controller = Controller(Config.app)
    controller.app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)