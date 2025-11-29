from typing import Callable
from dataclasses import dataclass


@dataclass
class Elo:
    """
    Representa uma cadeia de funções a serem executadas em sequência.
    Cada função recebe o resultado da anterior como entrada.
    """
    chain: list[Callable]
    
    def __init__(self, chain: list[Callable] = None):
        self.chain = chain if chain is not None else []
    
    
    def add(self, func: Callable) -> 'Elo':
        """Adiciona uma função à cadeia"""
        self.chain.append(func)
        return self
    
    
    def length(self) -> int:
        """Retorna o tamanho da cadeia"""
        return len(self.chain)
    
    
@dataclass
class Columns:
    """
    Representa uma coluna da tabela de dados.
    """