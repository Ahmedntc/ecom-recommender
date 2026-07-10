"""Preprocessamento dos dados do MovieLens.

Responsável por filtrar cold-start, normalizar ratings e codificar
IDs de usuários e filmes em índices sequenciais para uso nos modelos.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MovieLensPreprocessor:
    """Preprocessa os ratings explícitos do MovieLens (0.5 a 5.0).
    Normaliza os ratings para o intervalo [0, 1], remove usuários e
    filmes com poucas avaliações e codifica os IDs em índices sequenciais.
    
    Args:
        min_ratings_per_user: Remove usuários com poucas avaliações.
        min_ratings_per_movie: Remove filmes com poucas avaliações.
    """

    def __init__(
        self,
        min_ratings_per_user: int = 5,
        min_ratings_per_movie: int = 5,
    ) -> None:
        self._min_user = min_ratings_per_user
        self._min_movie = min_ratings_per_movie
        self._scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._user_para_idx: dict[int, int] = {}
        self._movie_para_idx: dict[int, int] = {}

    def fit(self, data: pd.DataFrame) -> "MovieLensPreprocessor":
        """Aprende a escala dos ratings e os mapeamentos de IDs.

        Args:
            data: DataFrame com colunas [userId, movieId, rating].

        Returns:
            Self.
        """
        filtrado = self._aplicar_filtro_frequencia(data)
        self._scaler.fit(filtrado[["rating"]].values.astype(np.float32))
        self._user_para_idx = {
            uid: idx for idx, uid in enumerate(sorted(filtrado["userId"].unique()))
        }
        self._movie_para_idx = {
            mid: idx for idx, mid in enumerate(sorted(filtrado["movieId"].unique()))
        }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normaliza os ratings e codifica os IDs.

        Args:
            data: DataFrame com colunas [userId, movieId, rating].

        Returns:
            DataFrame com colunas adicionais [user_idx, movie_idx, label].
        """
        resultado = self._aplicar_filtro_frequencia(data).copy()
        resultado["label"] = self._scaler.transform(
            resultado[["rating"]].values.astype(np.float32)
        ).flatten()
        resultado["user_idx"] = resultado["userId"].map(self._user_para_idx)
        resultado["movie_idx"] = resultado["movieId"].map(self._movie_para_idx)
        return resultado.dropna(subset=["user_idx", "movie_idx"]).reset_index(drop=True)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Executa fit e transform em uma única chamada.

        Args:
            data: DataFrame com colunas [userId, movieId, rating].

        Returns:
            DataFrame processado.
        """
        return self.fit(data).transform(data)

    def _aplicar_filtro_frequencia(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove usuários e filmes abaixo dos limiares de frequência."""
        contagem_users = data["userId"].value_counts()
        contagem_movies = data["movieId"].value_counts()
        users_validos = contagem_users[contagem_users >= self._min_user].index
        movies_validos = contagem_movies[contagem_movies >= self._min_movie].index
        mascara = (
            data["userId"].isin(users_validos)
            & data["movieId"].isin(movies_validos)
        )
        return data[mascara]

    @property
    def n_usuarios(self) -> int:
        """Número de usuários únicos após filtragem."""
        return len(self._user_para_idx)

    @property
    def n_filmes(self) -> int:
        """Número de filmes únicos após filtragem."""
        return len(self._movie_para_idx)