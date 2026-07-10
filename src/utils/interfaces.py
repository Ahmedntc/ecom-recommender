   
"""Interfaces abstratas do sistema de recomendação MovieLens.
Define os contratos que todas as implementações concretas devem seguir,
aplicando os princípios de Segregação de Interfaces e Inversão de
Dependências do SOLID.
"""
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd

# modelos de recomendação
class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, ratings: pd.DataFrame) -> "BaseRecommender":
        """Treina o modelo nos dados de avaliações do MovieLens.

        Args:
            ratings: DataFrame com colunas [userId, movieId, rating].

        Returns:
            Self.
        """

    @abstractmethod
    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Prediz scores de relevância para pares usuário-filme.
        Args:
            user_ids: Array de user_id.
            movie_ids: Array de id de filmes.

        Returns:
            Array de scores preditos, um para cada user-filme.
        """

    @abstractmethod
    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """Retorna os filmes recomendados para um usuário.

        Args:
            user_id: usuário alvo.
            top_k: num de recomendações a retornar.

        Returns:
             lista -> IDs de filmes (mais relevantes primeiro).
        """

#avaliador dos modelos
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model: BaseRecommender,
        test_data: pd.DataFrame,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Calcula métricas de avaliação para um modelo.

        Args:
            model: Modelo de recomendação treinado.
            test_data: Avaliações reais para comparação.
            top_k: Corte para métricas de ranking.

        Returns:
            Dicionário mapeando nome da métrica ao seu valor.
        """
