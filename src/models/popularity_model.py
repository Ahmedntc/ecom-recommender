#Baseline Recomenda os filmes mais populares com base na média ponderada de avaliações e contagem de avaliações independente do usuario
import numpy as np
import pandas as pd
from src.utils.interfaces import BaseRecommender

class PopularityRecommender(BaseRecommender):
    def __init__(self, min_ratings: int = 5) -> None:
        self._min_ratings = min_ratings
        self._ranked_movies: list[int] = []
        self._movie_scores: dict[int, float] = {}

    def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender":
        #usamos uma media ponderada para calcular a popularidade dos filmes, considerando tanto a média das avaliações quanto a quantidade de avaliações, 
        #para evitar que filmes com poucas avaliações tenham uma pontuação inflada.   
        stats = (
            ratings.groupby("movieId")["rating"]
            .agg(count="count", mean="mean")
            .query("count >= @self._min_ratings")
        )

        global_mean = ratings["rating"].mean()
        m = self._min_ratings

        stats["score"] = (
            stats["count"] / (stats["count"] + m) * stats["mean"]
            + m / (stats["count"] + m) * global_mean
        )

        stats = stats.sort_values("score", ascending=False)
        self._movie_scores = stats["score"].to_dict()
        self._ranked_movies = stats.index.tolist()
        return self

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        #ignoramos o user id, não é relevante para a nossa baseline ja que ela independe de usuário
  
        return np.array(
            [self._movie_scores.get(int(mid), 0.0) for mid in movie_ids],
            dtype=np.float32,
        )

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        return self._ranked_movies[:top_k]