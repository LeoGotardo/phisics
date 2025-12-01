import pandas as pd, numpy as np, matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from model.athleteModel import Athlete


class KNNModel:
    def __init__(self, k: int = 3):
        self.k = k
        self.athletes: list[Athlete] = []
    
    
    def fit(self, athletes: list[Athlete]) -> None:
        """Armazena os atletas de treinamento"""
        
        self.athletes = athletes
    
    
    def predict(self, athlete: Athlete) -> list[Athlete]:
        """Prediz os k atletas mais próximos"""
        
        distances = []
        for train_athlete in self.athletes:
            dist = athlete.distance_to(train_athlete)
            distances.append((dist, train_athlete))
        
        distances.sort(key=lambda x: x[0])
        neighbors = [athlete for _, athlete in distances[:self.k]]
        
        return neighbors
    
    def setupKNN(self, k: int) -> None:
        """Configura o valor de k"""
        
        self.k = k