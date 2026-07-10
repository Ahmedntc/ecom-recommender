"""Testes unitários para o MovieLensPreprocessor."""
import pandas as pd
import pytest
from src.data.preprocessors import MovieLensPreprocessor
@pytest.fixture()
def ratings_movielens() -> pd.DataFrame:
    """Fixture com dados no formato real do MovieLens com usuários e filmes suficientes."""
    linhas = []
    for user_id in range(1, 6):        # 5 usuários
        for movie_id in range(1, 11):  # 10 filmes
            linhas.append({
                "userId": user_id,
                "movieId": movie_id,
                "rating": float(user_id % 5 + 1),  # ratings variados entre 1.0 e 5.0
            })
    return pd.DataFrame(linhas)

class TestMovieLensPreprocessor:
    """Valida normalização, codificação e filtragem do preprocessador."""

    def test_fit_retorna_self(self, ratings_movielens: pd.DataFrame) -> None:
        """Deve retornar self para permitir encadeamento de métodos."""
        pre = MovieLensPreprocessor()
        resultado = pre.fit(ratings_movielens)
        assert resultado is pre

    def test_transform_adiciona_colunas_obrigatorias(
        self, ratings_movielens: pd.DataFrame
    ) -> None:
        """Deve adicionar as colunas user_idx, movie_idx e label."""
        pre = MovieLensPreprocessor()
        resultado = pre.fit_transform(ratings_movielens)
        assert {"user_idx", "movie_idx", "label"}.issubset(resultado.columns)

    def test_labels_normalizadas_entre_0_e_1(
        self, ratings_movielens: pd.DataFrame
    ) -> None:
        """Os labels devem estar no intervalo [0, 1] após normalização."""
        pre = MovieLensPreprocessor()
        resultado = pre.fit_transform(ratings_movielens)
        assert resultado["label"].between(0.0, 1.0).all()

    def test_n_usuarios_correto(self, ratings_movielens: pd.DataFrame) -> None:
        """Deve contar corretamente os usuários únicos após filtragem."""
        pre = MovieLensPreprocessor(min_ratings_per_user=5)
        pre.fit(ratings_movielens)
        assert pre.n_usuarios == 5

    def test_n_filmes_correto(self, ratings_movielens: pd.DataFrame) -> None:
        """Deve contar corretamente os filmes únicos após filtragem."""
        pre = MovieLensPreprocessor(min_ratings_per_movie=5)
        pre.fit(ratings_movielens)
        assert pre.n_filmes == 10

    def test_user_idx_sao_sequenciais(self, ratings_movielens: pd.DataFrame) -> None:
        """Os índices de usuários devem ser inteiros sequenciais a partir de 0."""
        pre = MovieLensPreprocessor()
        resultado = pre.fit_transform(ratings_movielens)
        indices = sorted(resultado["user_idx"].unique())
        assert indices == list(range(len(indices)))

    def test_movie_idx_sao_sequenciais(self, ratings_movielens: pd.DataFrame) -> None:
        """Os índices de filmes devem ser inteiros sequenciais a partir de 0."""
        pre = MovieLensPreprocessor()
        resultado = pre.fit_transform(ratings_movielens)
        indices = sorted(resultado["movie_idx"].unique())
        assert indices == list(range(len(indices)))

    def test_filtro_remove_usuarios_cold_start(self) -> None:
        """Usuários com poucas avaliações devem ser removidos."""
        data = pd.DataFrame({
            "userId":  [1, 1, 1, 1, 1, 99],  # usuário 99 tem apenas 1 avaliação
            "movieId": [1, 2, 3, 4, 5, 1],
            "rating":  [4, 3, 5, 2, 4, 3],
        })
        pre = MovieLensPreprocessor(min_ratings_per_user=5, min_ratings_per_movie=1)
        resultado = pre.fit_transform(data)
        assert 99 not in resultado["userId"].values

    def test_filtro_remove_filmes_cold_start(self) -> None:
        """Filmes com poucas avaliações devem ser removidos."""
        data = pd.DataFrame({
            "userId":  [1, 2, 3, 4, 5, 1],
            "movieId": [1, 1, 1, 1, 1, 99],  # filme 99 tem apenas 1 avaliação
            "rating":  [4, 3, 5, 2, 4, 3],
        })
        pre = MovieLensPreprocessor(min_ratings_per_user=1, min_ratings_per_movie=5)
        resultado = pre.fit_transform(data)
        assert 99 not in resultado["movieId"].values