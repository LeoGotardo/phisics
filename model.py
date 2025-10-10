import sqlalchemy

class Athlete(sqlalchemy.orm.Base):
    __tablename__ = 'athletes'
    id = ...
    

class Model:
    def __init__(self):
        self.engine = sqlalchemy.create_engine('sqlite:///db.sqlite3')
        self.session = sqlalchemy.orm.sessionmaker(bind=self.engine)()
        
        self.create_tables()