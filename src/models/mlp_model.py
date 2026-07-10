"""Modelo de recomendação baseado em MLP com embeddings (PyTorch).
Arquitetura :
    Embedding(userId) ──┐
                        ├─► concat ─► camadas MLP ─► sigmoid ─► score
    Embedding(movieId) ─┘
"""
import numpy as np
import pandas as pd
from src.utils.interfaces import BaseRecommender

class MLPRecommender(BaseRecommender):
    """Recomendador neural com filtragem colaborativa via MLP.
    Args:
        embedding_dim: Dimensão dos vetores de embedding de usuário e filme.
        hidden_dims: Número de neurônios em cada camada oculta do MLP.
        dropout: Probabilidade de dropout aplicada após cada camada oculta.
        learning_rate: Taxa de aprendizado do otimizador Adam.
        batch_size: Tamanho do mini-batch usado no treinamento.
        max_epochs: Número máximo de épocas de treinamento.
        early_stopping_patience: Para o treino se a val loss não melhorar
            por este número de épocas consecutivas.
        random_seed: Semente fixa para garantir reprodutibilidade.
    """
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        random_seed: int = 21,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed
        self._network = None  # torch.nn.Module — inicializado no fit()

    def fit(self, ratings: pd.DataFrame) -> "MLPRecommender":
        raise NotImplementedError("Ainda nao")

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Ainda nao")

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        raise NotImplementedError("Ainda nao")