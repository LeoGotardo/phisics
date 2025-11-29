import pandas as pd
from typing import Callable, Any
from io import BytesIO

from utils.dataclasses import Columns, Elo
from model.athleteModel import Athlete
from utils.dataUtils import DataUtils
from config import Config


class CSVImportElo(Elo):
    """
    Pipeline para importação e validação de arquivos CSV com dados de atletas.
    Herda de Elo e implementa a cadeia de processamento.
    """
    
    
    def __init__(self) -> None:
        # Inicializar a cadeia vazia (do Elo pai)
        super().__init__()
        
        # Configurar colunas esperadas
        self.COLUMNS: list[Columns] = Config.COLUMNS
        
        # Construir a cadeia de processamento
        self._build_chain()
    
    
    def _build_chain(self) -> None:
        """Constrói a cadeia de funções para processar o CSV"""
        self.chain = [
            self.renderCSV,
            self.validateColumns,
            self.sanitizeCSV,
            self.listAthletes
        ]
    
    
    def renderCSV(self, csvFile: bytes) -> pd.DataFrame:
        """
        Lê o arquivo CSV e retorna um DataFrame.
        
        Args:
            csvFile: Bytes do arquivo CSV ou objeto file-like
            
        Returns:
            DataFrame com os dados do CSV
        """
        # Se for bytes, converter para BytesIO
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
            # Normalizar nomes das colunas para lowercase
            frame_columns = [col.lower() for col in data.columns]
            
            # Verificar se todas as colunas esperadas existem
            missing_columns = []
            for column in self.COLUMNS:
                column_name = column['name'].lower()
                if column_name not in frame_columns:
                    missing_columns.append(column['name'])
            
            if missing_columns:
                raise ValueError(f"Colunas faltando no CSV: {', '.join(missing_columns)}")
            
            # Normalizar nomes das colunas no DataFrame
            data.columns = [col.lower() for col in data.columns]
            
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
        data = data.copy()  # Trabalhar com cópia para não modificar original
        
        for column_config in self.COLUMNS:
            column_name = column_config['name'].lower()
            column_type = column_config['type']
            
            # Verificar se a coluna existe
            if column_name not in data.columns:
                continue
            
            # Aplicar validação conforme o tipo
            try:
                if column_type == 'num':
                    data[column_name] = DataUtils.validNum(data[column_name])
                    
                elif column_type == 'date':
                    data[column_name] = DataUtils.validDate(data[column_name])
                    
                elif column_type == 'sex':
                    data[column_name] = DataUtils.validSex(data[column_name])
                    
                elif column_type == 'string':
                    data[column_name] = DataUtils.validString(data[column_name])
                    
                else:
                    raise ValueError(f"Tipo de coluna desconhecido: {column_type}")
                    
            except Exception as e:
                raise ValueError(f"Erro ao sanitizar coluna '{column_name}': {str(e)}")
        
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
                    data_nascimento=row['data_nascimento'],
                    sexo=row['sexo'],
                    estatura=row['estatura'],
                    envergadura=row['envergadura'],
                    arremesso=row['arremesso'],
                    salto_horizontal=row['salto_horizontal'],
                    abdominais=row['abdominais']
                )
                athletes.append(athlete)
                
            except KeyError as e:
                raise ValueError(f"Coluna faltando na linha {index}: {str(e)}")
            except Exception as e:
                raise ValueError(f"Erro ao criar atleta na linha {index}: {str(e)}")
        
        return athletes