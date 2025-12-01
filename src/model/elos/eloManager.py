from typing import Callable, Any, Optional
from src.utils.dataclasses import Elo


class EloManager:
    """
    Gerencia a execução de uma cadeia de funções (pipeline).
    Cada função na cadeia recebe o resultado da anterior.
    """
    
    def __init__(self, elo: Elo):
        self.elo: Elo = elo
        self.chain: list[Callable] = self.elo.chain.copy()
        self.current_index: int = 0
        self.result: Any = None
        self.history: list[Any] = []
        self.is_completed: bool = False
        self.error: Optional[Exception] = None
    
    
    def startElo(self, initial_value: Any = None) -> Any:
        """
        Inicia a execução da cadeia completa.
        
        Args:
            initial_value: Valor inicial a ser passado para a primeira função
            
        Returns:
            O resultado final da cadeia
        """
        
        self.result = initial_value
        self.current_index = 0
        self.history = [initial_value]
        self.is_completed = False
        self.error = None
        
        while not self.isDone():
            try:
                self.nextNode()
            except Exception as e:
                self.error = e
                raise(e)
        
        return self.returnResult()
    
    
    def nextNode(self) -> Any:
        """
        Executa o próximo nó (função) na cadeia.
        
        Returns:
            O resultado da função executada
        """
        
        if self.isDone():
            return self.result
        
        current_func = self.chain[self.current_index]
        
        try:
            self.result = current_func(self.result)
            
            self.history.append(self.result)
            
            self.current_index += 1
            
            if self.current_index >= len(self.chain):
                self.is_completed = True
            
            return self.result
            
        except Exception as e:
            self.error = e
            self.is_completed = True
            raise
    
    
    def isDone(self) -> bool:
        """
        Verifica se a cadeia foi completamente executada.
        
        Returns:
            True se completou, False caso contrário
        """
        
        return self.current_index >= len(self.chain) or self.is_completed
    
    
    def returnResult(self) -> Any:
        """
        Retorna o resultado final da cadeia.
        
        Returns:
            O resultado da última função executada
        """
        
        return self.result
    
    
    def getHistory(self) -> list[Any]:
        """
        Retorna o histórico de todos os resultados intermediários.
        
        Returns:
            Lista com todos os resultados (incluindo inicial e intermediários)
        """
        
        return self.history
    
    
    def reset(self):
        """Reseta o manager para executar novamente"""
        
        self.current_index = 0
        self.result = None
        self.history = []
        self.is_completed = False
        self.error = None
    
    
    def getCurrentStep(self) -> int:
        """Retorna o índice do passo atual"""
        
        return self.current_index
    
    
    def hasError(self) -> bool:
        """Verifica se houve erro na execução"""
        
        return self.error is not None