from flask import Flask, render_template, request, redirect, url_for, flash, Blueprint, send_file
from controller.controller import Controller
from models.model import Config
from io import BytesIO

import traceback, sys, json


class View:
    def __init__(self, app):
        self.app: Flask = app
        self.controller = Controller(Config.app)
        self.controller = Controller(self.app)
        self.blueprint = Blueprint('view', __name__, template_folder='templates')
        
        self.app.register_blueprint(self.blueprint)
        self.app.register_error_handler(Exception, self.handleException)
        self.defineRouters()
        
        self.app.run(debug=True)
        
        
    def defineRouters(self):
        self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
        self.app.add_url_rule('/analise', view_func=self.review, methods=['GET', 'POST'])
        self.app.add_url_rule('/createAthlete', view_func=self.createAthlete, methods=['GET', 'POST'])
        self.app.add_url_rule('/viewData', view_func=self.viewPage, methods=['GET'])
        self.app.add_url_rule('/loadCSVData', view_func=self.loadCSVData, methods=['POST'])
        self.app.add_url_rule('/exportData', view_func=self.exportData, methods=['GET'])
    
    
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
        status, info = self.controller.getDashboardInfo()
        if status == True:
            return render_template('dashboard.html')
        if status == -1:
            raise Exception(info)
        
        
    def loadCSVData(self):
        if request.method == 'POST':
            status, response = self.controller.loadCSVData(request.files['csvFile'])
            if status == -1:
                raise Exception(response)
            if status == True:
                flash(response, category='success')
            else:
                flash(response, category='error')
            return render_template('cadastro.html')
        else:
            return render_template('404.html')
    
    
    def exportData(self):
        if request.method == 'GET':
            status, info = self.controller.exportData()
            if status == True:
                return send_file(BytesIO(info), as_attachment=True, attachment_filename='data.csv')
            if status == -1:
                raise Exception(info)
            else:
                flash(info, category='error')
                return render_template('exportData.html')
        else:
            return render_template('404.html')
        
    
    def createAthlete(self):
        match request.method:
            case 'POST':
                status, response = self.controller.putAthlete(request.form)
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
            case 'POST':
                status, response = self.controller.getAthlets(request.form)
                if status == -1:
                    raise Exception(response)
                if status == True:
                    flash(response, category='success')
                else:
                    flash(response, category='error')
                return render_template('analise.html')
            case "GET":
                return render_template('analise.html')
            case _:
                return render_template('404.html')


    def viewPage(self):
        status, info = self.controller.getViewInfo()
        if status == True:
            return render_template('view.html', info)
        if status == -1:
            raise Exception(info)
        else:
            flash(info, category='error')
            return render_template('view.html')
        
        
if __name__ == '__main__':
    View(Config.app)