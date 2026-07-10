#Testes unitários para o ModelFactory (padrão Factory).
import pytest
from src.models.mlp_model import MLPRecommender
from src.models.model_factory import ModelFactory, ModelType
from src.models.popularity_model import PopularityRecommender

class TestModelFactory:
    def test_cria_modelo_popularity(self) -> None:
        """Deve retornar uma instância de PopularityRecommender."""
        factory = ModelFactory()
        modelo = factory.create(ModelType.POPULARITY)
        assert isinstance(modelo, PopularityRecommender)

    def test_cria_modelo_mlp(self) -> None:
        """Deve retornar uma instância de MLPRecommender."""
        factory = ModelFactory()
        modelo = factory.create(ModelType.MLP)
        assert isinstance(modelo, MLPRecommender)

    def test_tipo_desconhecido_lanca_value_error(self) -> None:
        """Deve lançar ValueError para tipos não registrados."""
        factory = ModelFactory()
        with pytest.raises(ValueError, match="não encontrado"):
            factory.create("modelo_inexistente")

    def test_modelos_disponiveis_contem_defaults(self) -> None:
        """Deve listar os modelos registrados por padrão."""
        factory = ModelFactory()
        assert ModelType.POPULARITY in factory.modelos_disponiveis
        assert ModelType.MLP in factory.modelos_disponiveis

    def test_registra_modelo_customizado(self) -> None:
        """Deve permitir registrar e criar um modelo externo."""
        factory = ModelFactory()
        factory.register("custom", PopularityRecommender)
        modelo = factory.create("custom")
        assert isinstance(modelo, PopularityRecommender)

    def test_kwargs_sao_repassados_ao_modelo(self) -> None:
        """Deve repassar argumentos ao construtor do modelo."""
        factory = ModelFactory()
        modelo = factory.create(ModelType.POPULARITY, min_ratings=10)
        assert isinstance(modelo, PopularityRecommender)
        assert modelo._min_ratings == 10