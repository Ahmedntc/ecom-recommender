"""Testes unitários para o PopularityRecommender."""
import numpy as np
import pandas as pd
import pytest
from src.models.popularity_model import PopularityRecommender
@pytest.fixture()
def ratings_movielens() -> pd.DataFrame:
    """Fixture com dados no formato real do MovieLens."""
    return pd.DataFrame(
        {
            "userId":  [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "movieId": [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4],
            "rating":  [5, 3, 4, 4, 5, 2, 3, 4, 5, 2, 3, 4],
        }
    )

class TestPopularityRecommender:
    def test_fit_retorna_self(self, ratings_movielens: pd.DataFrame) -> None:
        """Deve retornar self para permitir encadeamento de métodos."""
        modelo = PopularityRecommender(min_ratings=1)
        resultado = modelo.fit(ratings_movielens)
        assert resultado is modelo

    def test_recommend_respeita_top_k(self, ratings_movielens: pd.DataFrame) -> None:
        """Deve retornar exatamente top_k filmes."""
        modelo = PopularityRecommender(min_ratings=1)
        modelo.fit(ratings_movielens)
        recomendacoes = modelo.recommend(user_id=1, top_k=2)
        assert len(recomendacoes) == 2

    def test_recommend_retorna_ids_validos(self, ratings_movielens: pd.DataFrame) -> None:
        """Deve retornar apenas IDs de filmes presentes nos dados."""
        modelo = PopularityRecommender(min_ratings=1)
        modelo.fit(ratings_movielens)
        filmes_conhecidos = set(ratings_movielens["movieId"].unique())
        recomendacoes = modelo.recommend(user_id=1, top_k=3)
        assert all(mid in filmes_conhecidos for mid in recomendacoes)

    def test_predict_retorna_scores_positivos(self, ratings_movielens: pd.DataFrame) -> None:
        """Deve retornar scores maiores ou iguais a zero."""
        modelo = PopularityRecommender(min_ratings=1)
        modelo.fit(ratings_movielens)
        scores = modelo.predict(
            user_ids=np.array([1, 2]),
            movie_ids=np.array([1, 2]),
        )
        assert all(s >= 0.0 for s in scores)

    def test_predict_filme_desconhecido_retorna_zero(
        self, ratings_movielens: pd.DataFrame
    ) -> None:
        """Deve retornar 0.0 para filmes não vistos no treino."""
        modelo = PopularityRecommender(min_ratings=1)
        modelo.fit(ratings_movielens)
        scores = modelo.predict(
            user_ids=np.array([1]),
            movie_ids=np.array([9999]),
        )
        assert scores[0] == 0.0

    def test_min_ratings_filtra_filmes_com_poucas_avaliacoes(
        self, ratings_movielens: pd.DataFrame
    ) -> None:
        """Nenhum filme deve ser rankeado quando min_ratings é muito alto."""
        modelo = PopularityRecommender(min_ratings=100)
        modelo.fit(ratings_movielens)
        assert modelo._ranked_movies == []

    def test_recommend_top_k_maior_que_catalogo(
        self, ratings_movielens: pd.DataFrame
    ) -> None:
        """Não deve estourar quando top_k é maior que o catálogo disponível."""
        modelo = PopularityRecommender(min_ratings=1)
        modelo.fit(ratings_movielens)
        recomendacoes = modelo.recommend(user_id=1, top_k=9999)
        assert len(recomendacoes) <= ratings_movielens["movieId"].nunique()