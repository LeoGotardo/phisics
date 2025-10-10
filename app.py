from flask import Flask, render_template, request, redirect, url_for, flash, Blueprint
from controller import Controller

class view:
    def __init__(self, app):
        self.app = app
        self.controller = Controller()
        self.blueprint = Blueprint('view', __name__, template_folder='templates')
        
        self.app.register_blueprint(self.blueprint)
        self.defineRouters()
        
    def defineRouters(self):
        self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
        self.app.add_url_rule('/analise', view_func=self.review, methods=['GET', 'POST'])
        self.app.add_url_rule('/createAthlete', view_func=self.createAthlete, methods=['GET', 'POST'])
        self.app.add_url_rule('/viewData', view_func=self.viewPage, methods=['GET'])
    
    
    def index(self):
        status, info = self.controller.getDashboardInfo()
        if status == True:
            return render_template('dashboard.html')
        if status == -1:
            return render_template('errorPage.html', **info)
    
    
    def createAthlete(self):
        match request.method:
            case 'POST':
                status, response = self.controller.putAthlete(request.form)
                if status == -1:
                    return render_template('errorPage.html', **response)
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
                status, response = self.controller.createCluster(request.form)
                if status == -1:
                    return render_template('errorPage.html', **response)
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
            return render_template('view.html', **info)
        if status == -1:
            return render_template('errorPage.html', **info)
        else:
            flash(info, category='error')
            return render_template('view.html')