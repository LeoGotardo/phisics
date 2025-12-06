import re, pandas as pd

from datetime import datetime


class DataUtils:
    """
    Classe utilitária para validação e sanitização de dados.
    Fornece métodos estáticos para validar diferentes tipos de dados.
    """
    
    STRING_REGEX = re.compile(r'^[a-zA-ZÀ-ÿ\s]+$')
    
    DATE_FORMATS = [
        '%Y-%m-%d',      # 2024-01-15
        '%d/%m/%Y',      # 15/01/2024
        '%d-%m-%Y',      # 15-01-2024
        '%Y/%m/%d',      # 2024/01/15
    ]
    
    
    @staticmethod
    def validNum(dataFrame: pd.DataFrame, columnName: str) -> pd.DataFrame | bool:
        """
        Valida e converte um valor para número.
        
        Args:
            x: Colouna a ser validada
            
        Returns:
            dataFrame: DataFrame com a coluna convertida
            bool: False se não for possível converter'
        """
        
        try:
            dataFrame[columnName] = pd.to_numeric(dataFrame[columnName], downcast='float')
            
            return dataFrame
            
        except (ValueError, TypeError, AttributeError):
            return False
    
    
    @staticmethod
    def validDate(dataFrame: pd.DataFrame, columnName: str) -> datetime | bool:
        """
        Valida e converte um valor para data.
        
        Args:
            dataFrame: Dataframe a ser validado
            columnName: Nome da coluna a ser validada
            
        Returns:
            datetime: Data convertida
            bool: False se não for possível converter
        """
        try:
            dataFrame[columnName] = pd.to_datetime(dataFrame[columnName])
            
            return dataFrame
            
        except (ValueError, TypeError, AttributeError):
            return False
    
    
    @staticmethod
    def validSex(dataFrame: pd.DataFrame, columnName: str ) -> int | bool:
        """
        Valida e converte um valor para sexo.
        
        Args:
            dataFrame: DataFrame a ser validado
            columnName: Nome da coluna a ser validada
            
        Returns:
            int: 0 para masculino, 1 para feminino
            bool: False se não for válido
        """
            
        try:
            def convert(x):
                sexStr = str(x).strip().upper()
                
                maleValues = ['M', 'MASCULINO', 'MASC', 'MALE', '0']
                femaleValues = ['F', 'FEMININO', 'FEM', 'FEMALE', '1']
                
                if sexStr in maleValues:
                    return 0
                elif sexStr in femaleValues:
                    return 1
                else:
                    raise ValueError(f"Valor '{sexStr}' não é válido para sexo")
                
            dataFrame[columnName] = dataFrame[columnName].apply(convert)
            
            return dataFrame
                
        except (ValueError, TypeError, AttributeError):
            return False
        
        
    def validStr(dataFrame: pd.DataFrame, columnName: str, minLength: int = 1, regexFilter = STRING_REGEX, maxLength: int = 200) -> str | bool:
        def convert(x):
            strValue = str(x).strip()
            
            if len(strValue) < minLength or len(strValue) > maxLength:
                raise ValueError(f"String '{strValue}' não atende aos requisitos de comprimento")
            
            if not regexFilter.match(strValue):
                raise ValueError(f"String '{strValue}' não atende ao filtro regex")
            
            return strValue
        try:
            dataFrame[columnName] = dataFrame[columnName].apply(convert)
            
            return dataFrame
        except (ValueError, TypeError, AttributeError):
            return False