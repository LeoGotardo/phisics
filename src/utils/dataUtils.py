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
    def validNum(x) -> float | bool:
        """
        Valida e converte um valor para número.
        
        Args:
            x: Valor a ser validado
            
        Returns:
            float: Número convertido
            bool: False se não for possível converter
        """
        
        try:
            if isinstance(x, str):
                x = x.strip()
                x = x.replace(',', '.')
            
            num = float(x)
            
            if pd.isna(num) or not pd.notna(num):
                return False
                
            return num
            
        except (ValueError, TypeError, AttributeError):
            return False
    
    
    @staticmethod
    def validDate(x, strict: bool = False) -> datetime | bool:
        """
        Valida e converte um valor para data.
        
        Args:
            x: Valor a ser validado
            strict: Se True, aceita apenas formato YYYY-MM-DD
            
        Returns:
            datetime: Data convertida
            bool: False se não for possível converter
        """
        
        if pd.isna(x):
            return False
            
        try:
            if isinstance(x, datetime):
                return x
            
            dateStr = str(x).strip()
            
            formats = [DataUtils.DATE_FORMATS[0]] if strict else DataUtils.DATE_FORMATS
            
            for fmt in formats:
                try:
                    return datetime.strptime(dateStr, fmt)
                except ValueError:
                    continue
            
            return False
            
        except (ValueError, TypeError, AttributeError):
            return False
    
    
    @staticmethod
    def validSex(x) -> int | bool:
        """
        Valida e converte um valor para sexo.
        
        Args:
            x: Valor a ser validado ('M', 'F', 'Masculino', 'Feminino', etc)
            
        Returns:
            int: 0 para masculino, 1 para feminino
            bool: False se não for válido
        """
        
        if pd.isna(x):
            return False
            
        try:
            sexStr = str(x).strip().upper()
            
            maleValues = ['M', 'MASCULINO', 'MASC', 'MALE', '0']
            femaleValues = ['F', 'FEMININO', 'FEM', 'FEMALE', '1']
            
            if sexStr in maleValues:
                return 0
            elif sexStr in femaleValues:
                return 1
            else:
                return False
                
        except (ValueError, TypeError, AttributeError):
            return False
    
    
    @staticmethod
    def validString(x, minLength: int = 1, regexFilter = STRING_REGEX, maxLength: int = 200) -> str | bool:
        """
        Valida e sanitiza uma string.
        
        Args:
            x: Valor a ser validado
            min_length: Tamanho mínimo da string
            max_length: Tamanho máximo da string
            
        Returns:
            str: String validada e sanitizada
            bool: False se não for válida
        """
        
        if pd.isna(x):
            return False
            
        try:
            text = str(x).strip()
            
            if len(text) < minLength or len(text) > maxLength:
                return False
            
            if not regexFilter.match(text):
                return False
            
            text = ' '.join(text.split())
            
            text = text.title()
            
            return text
            
        except (ValueError, TypeError, AttributeError):
            return False