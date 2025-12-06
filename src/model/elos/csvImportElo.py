import pandas as pd

from src.utils.dataclasses import Column, Elo
from src.model.athleteModel import Athlete
from src.utils.dataUtils import DataUtils
from src.config import Config
from io import BytesIO
from icecream import ic


class CSVImportElo(Elo):
    """
    Pipeline para importação e validação de arquivos CSV com dados de atletas.
    Herda de Elo e implementa a cadeia de processamento.
    """
    
    def __init__(self) -> None:        
        super().__init__()
        
        self.COLUMNS: list[Column] = Config.COLUMNS.copy()
        
        self._build_chain()
    
    
    def _build_chain(self) -> None:
        """Constrói a cadeia de funções para processar o CSV"""
        
        self.chain = [
            self.renderCSV,
            self.validateColumns,
            self.sanitizeCSV,
            self.listAthletes,
        ]
    
    
    def renderCSV(self, csvFile: bytes) -> pd.DataFrame:
        """
        Lê o arquivo CSV e retorna um DataFrame.
        
        Args:
            csvFile: Bytes do arquivo CSV ou objeto file-like
            
        Returns:
            DataFrame com os dados do CSV
        """
        
        
        if isinstance(csvFile, bytes):
            csvFile = BytesIO(csvFile)
        
        return pd.read_csv(csvFile)
    
    
    def validateColumns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Valida se todas as colunas esperadas existem no DataFrame.
        
        Args:
            data: DataFrame a ser validado
            
        Returns:
            O mesmo DataFrame se válido
            
        Raises:
            ValueError: Se alguma coluna estiver faltando
        """
        
        try:
            frameColumns = data.columns.tolist()
            
            missingColumns = []
            for column in self.COLUMNS:
                columnName = column.name
                if columnName not in frameColumns:
                    missingColumns.append(column.name)
            
            if missingColumns:
                raise ValueError(f"Colunas faltando no CSV: {', '.join(missingColumns)}")
            
            return data
            
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Erro ao validar colunas: {str(e)}")
    
    
    def sanitizeCSV(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitiza e valida os dados de cada coluna conforme seu tipo.
        
        Args:
            data: DataFrame com dados brutos
            
        Returns:
            DataFrame com dados sanitizados e validados
        """
        

        data = data.copy()
        
        for columnConfig in self.COLUMNS:
            columnName = columnConfig.name
            columnType = columnConfig.type
            
            if columnName not in data.columns:
                continue
            
            try:
                if columnType == 'num':
                    data = DataUtils.validNum(data, columnName)   
                elif columnType == 'date':
                    data = DataUtils.validDate(data, columnName) 
                elif columnType == 'sex':
                    data = DataUtils.validSex(data, columnName) 
                elif columnType == 'string':
                    data = DataUtils.validStr(data, columnName)
                else:
                    raise ValueError(f"Tipo de coluna desconhecido: {columnType}")    
            except Exception as e:
                raise ValueError(f"Erro ao sanitizar coluna '{columnName}': {str(e)}")
        
        return data
    
    
    def listAthletes(self, data: pd.DataFrame) -> list[Athlete]:
        """
        Converte as linhas do DataFrame em objetos Athlete.
        
        Args:
            data: DataFrame com dados sanitizados
            
        Returns:
            Lista de objetos Athlete
        """
        athletes: list[Athlete] = []
        
        for index, row in data.iterrows():
            try:
                athlete = Athlete(
                    nome=row['nome'],
                    dataNascimento=row['dataNascimento'],
                    sexo=row['sexo'],
                    altura=row['altura'],
                    envergadura=row['envergadura'],
                    arremesso=row['arremesso'],
                    saltoHorizontal=row['saltoHorizontal'],
                    abdominais=row['abdominais']
                )
                athletes.append(athlete)
                
            except KeyError as e:
                raise ValueError(f"Coluna faltando na linha {index}: {str(e)}")
            except Exception as e:
                raise ValueError(f"Erro ao criar atleta na linha {index}: {str(e)}")
        
        return athletes