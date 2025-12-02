from flask import Flask, render_template, request, redirect, url_for, flash, Blueprint, send_file
from src.model.model import Model
from icecream import ic
from io import BytesIO
from src.config import Config
import traceback, sys, json


class Controller:
    def __init__(self, app):
        self.app: Flask = app
        self.model = Model()
        self.blueprint = Blueprint('view', __name__, template_folder='templates')
        
        self.app.register_blueprint(self.blueprint)
        self.app.register_error_handler(Exception, self.handleException)
        self.defineRouters()
        
        self.app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
        
        
    def defineRouters(self):
        self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
        self.app.add_url_rule('/analise', view_func=self.review, methods=['GET', 'POST'])
        self.app.add_url_rule('/cadastro', view_func=self.createAthlete, methods=['GET', 'POST'])
        self.app.add_url_rule('/view', view_func=self.viewPage, methods=['GET'])
        self.app.add_url_rule('/loadCSVData', view_func=self.loadCSVData, methods=['POST'])
        self.app.add_url_rule('/exportData', view_func=self.exportData, methods=['GET'])
        
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
            ic(info)
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
                ic(error_msg)
                flash(f'Erro ao importar CSV: {error_msg}', category='error')
                return redirect(url_for('renderCadastro'))
        else:
            return render_template('404.html'), 404
    
    
    def exportData(self):
        """
        Exporta dados dos atletas em formato ZIP com CSV e gráficos.
        """
        if request.method == 'GET':
            try:
                # Parâmetros opcionais da URL
                full_data = request.args.get('full_data', 'true').lower() == 'true'
                athletes_ids = request.args.get('ids', None)
                
                if athletes_ids:
                    athletes_ids = athletes_ids.split(',')
                
                # Exportar dados através do model
                status, result = self.model.exportData(
                    fullData=full_data,
                    athletesIds=athletes_ids
                )
                
                if status == True:
                    # result contém os bytes do arquivo ZIP
                    return send_file(
                        BytesIO(result),
                        mimetype='application/zip',
                        as_attachment=True,
                        download_name='talent_scout_export.zip'
                    )
                    
                elif status == False:
                    flash(result, category='warning')
                    return redirect(url_for('renderViewPage'))
                    
                else:  # status == -1
                    raise Exception(result)
                    
            except Exception as e:
                error_msg = f'{type(e).__name__}: {e}'
                ic(error_msg)
                flash(f'Erro ao exportar dados: {error_msg}', category='error')
                return redirect(url_for('renderViewPage'))
        else:
            return render_template('404.html'), 404
        
    
    def createAthlete(self):
        match request.method:
            case 'POST':
                ic(request.form.listvalues())
                dataDict = {
                'nome' : request.form.get('nome'),
                'dataNascimento' : request.form.get('dataNascimento'),
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
        match request.method:
            case 'GET':
                success, athletes = self.model.getAthletes(paginated=True)
                if success:
                    ic(athletes)
                    return render_template('analise.html', athletes=athletes)
                else:
                    raise Exception(athletes)
            case "POST":
                sort = request.form.get('sort', 'name')
                sortOrder = request.form.get('sortOrder', 'desc')
                query = request.form.get('query', '')
                paginated = request.form.get('paginated', 'false') == 'true'
                page = request.form.get('page', '1')
                perPage = request.form.get('perPage', '10')
                
                success, athletes = self.model.getAthletes(sort, sortOrder, query, page, perPage, paginated=True)
                if success == True:
                    return render_template('analise.html', athletes=athletes)
                else:
                    raise Exception(athletes)
            case _:
                return render_template('404.html')


    def viewPage(self):
        """
        Renderiza a página de visualização com dados estatísticos.
        Gera gráficos PCA, correlação, core strength, radar e métricas de qualidade.
        """
        match request.method:
            case 'GET':
                success, info = self.model.getViewInfo()
                
                if success == True:
                    return render_template('view.html', info=info)
                    
                elif success == False:
                    flash(info, category='warning')
                    
                    empty_data = {
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
                    
                    return render_template('view.html', info=empty_data)
                
                else:
                    raise Exception(info)
            
            case _:
                return render_template('404.html'), 404
        
        
if __name__ == '__main__':
    controller = Controller(Config.app)
    controller.app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)