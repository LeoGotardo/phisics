from datetime import datetime


class DataUtils:
    def __init__(self) -> None:
        self.stringRegex =
    
    def validNum(self, x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return False
    
    
    def validDate(self, x):
        try:
            return datetime.strptime(str(x), '%Y-%m-%d')
        except (ValueError, TypeError):
            return False
    
    
    def validSex(self, x):
        if str(x).upper() == 'M':
            return 0
        if str(x).upper() == 'F':
            return 1
        return False
    
    
    def validString(self, x):
        try:
            return str(x)
        except (ValueError, TypeError):
            return False
        