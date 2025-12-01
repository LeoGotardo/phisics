from src.model.elos.eloManager import EloManager
from typing import Callable, Any, Literal
from dataclasses import dataclass, field



@dataclass
class Elo:
    """
    Representa uma cadeia de funções a serem executadas em sequência.
    Cada função recebe o resultado da anterior como entrada.
    
    Attributes:
        chain: Lista de funções na cadeia
    """
    
    chain: list[Callable] = field(default_factory=list)
    
    
    def add(self, func: Callable) -> 'Elo':
        """Adiciona uma função à cadeia"""
        
        self.chain.append(func)
        return self
    
    
    def length(self) -> int:
        """Retorna o tamanho da cadeia"""
        
        return len(self.chain)
    
    
    def clear(self) -> 'Elo':
        """Limpa a cadeia"""
        
        self.chain.clear()
        return self
    
    
    def remove(self, func: Callable) -> 'Elo':
        """Remove uma função específica da cadeia"""
        
        if func in self.chain:
            self.chain.remove(func)
        return self
    
    
    def startFrom(self, func: Callable, *args, **kwargs) -> any:
        """Inicia a cadeia a partir de uma função específica"""
        for i, func in enumerate(self.chain):
            if func == func:
                self.chain = self.chain[i:]
        
        return EloManager(self).startElo(*args, **kwargs)


@dataclass
class Column:
    """
    Representa uma coluna da tabela de dados com suas configurações de validação.
    
    Attributes:
        name: Nome da coluna
        type: Tipo de dado ('num', 'date', 'sex', 'string')
        required: Se a coluna é obrigatória
        minValue: Valor mínimo (para números)
        maxValue: Valor máximo (para números)
        minLength: Tamanho mínimo (para strings)
        maxLength: Tamanho máximo (para strings)
        defaultValue: Valor padrão caso esteja vazio
        description: Descrição da coluna
        validator: Função customizada de validação
    """
    
    name: str
    type: Literal['num', 'date', 'sex', 'string']
    required: bool = True
    minValue: float | None = None
    maxValue: float | None = None
    minLength: int | None = None
    maxLength: int | None = None
    defaultValue: Any = None
    description: str = ""
    validator: Callable[[Any], bool] | None = None
    
    
    def toDict(self) -> dict:
        """Converte a coluna para dicionário"""
        
        return {
            'name': self.name,
            'type': self.type,
            'required': self.required,
            'minValue': self.minValue,
            'maxValue': self.maxValue,
            'minLength': self.minLength,
            'maxLength': self.maxLength,
            'defaultValue': self.defaultValue,
            'description': self.description
        }
    
    
    @classmethod
    def fromDict(cls, data: dict) -> 'Column':
        """Cria uma instância a partir de um dicionário"""
        
        return cls(
            name=data['name'],
            type=data['type'],
            required=data.get('required', True),
            minValue=data.get('minValue'),
            maxValue=data.get('maxValue'),
            minLength=data.get('minLength'),
            maxLength=data.get('maxLength'),
            defaultValue=data.get('defaultValue'),
            description=data.get('description', '')
        )