"""Factory de modelos de recomendação — padrão de projeto criacional
Exemplo de uso:
    >>> factory = ModelFactory()
    >>> model = factory.create(ModelType.POPULARITY)
    >>> model.fit(ratings_df)
"""
from enum import StrEnum
from typing import Any
from src.utils.interfaces import BaseRecommender
#enum modelos disponiveis
class ModelType(StrEnum):
    POPULARITY = "baseline"
    MLP = "mlp"

class ModelFactory:
    def __init__(self) -> None:
        # Dicionário interno que mapeia tipo → classe concreta
        self._registry: dict[str, type[BaseRecommender]] = {}
        self._registrar_modelos_padrao()

    def _registrar_modelos_padrao(self) -> None:
        from src.models.mlp_model import MLPRecommender
        from src.models.popularity_model import PopularityRecommender

        self._registry[ModelType.POPULARITY] = PopularityRecommender
        self._registry[ModelType.MLP] = MLPRecommender

    def register(self, model_type: str, model_class: type[BaseRecommender]) -> None:
        """Registra um novo tipo de modelo no factory.
        Args:
            model_type: Chave string usada para solicitar este modelo.
            model_class: Classe concreta que implementa BaseRecommender.
        """
        self._registry[model_type] = model_class

    def create(self, model_type: str, **kwargs: Any) -> BaseRecommender:
        """Instancia e retorna um modelo do tipo solicitado.
        Args:
            model_type: Chave que identifica qual modelo criar.
            **kwargs: Argumentos repassados ao construtor do modelo.

        Returns:
            Instância do modelo escolhido.

        Raises:
            ValueError: Quando o tipo solicitado não está registrado.
        """
        model_class = self._registry.get(model_type)

        if model_class is None:
            disponiveis = ", ".join(self._registry)
            raise ValueError(
                f"Tipo de modelo '{model_type}' não encontrado. "
                f"Disponíveis: {disponiveis}"
            )

        return model_class(**kwargs)

    @property
    def modelos_disponiveis(self) -> list[str]:
        """Retorna todas as chaves de modelos registrados."""
        return list(self._registry)