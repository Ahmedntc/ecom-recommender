"""
Configs carregadas de variáveis de ambiente. 
CONFIGURAÇÕES SÃO CARREGADAS AUTOMATICAMENTE DO ARQUIVO .env,
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class MLflowSettings(BaseSettings):
    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_name: str = Field(default="movielens-recommender")
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

class ModelSettings(BaseSettings):
#hiperparametros
    embedding_dim: int = Field(default=64, gt=0)
    hidden_dims: list[int] = Field(default=[256, 128, 64])
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    batch_size: int = Field(default=512, gt=0)
    max_epochs: int = Field(default=50, gt=0)
    early_stopping_patience: int = Field(default=5, gt=0)
    random_seed: int = Field(default=21)
    model_config = SettingsConfigDict(env_prefix="MODEL_")

class DataSettings(BaseSettings):
#caminhos do DataSet e filtragens
    raw_path: Path = Field(default=Path("data/raw"))
    processed_path: Path = Field(default=Path("data/processed"))
    min_ratings_per_user: int = Field(default=5, gt=0)
    min_ratings_per_movie: int = Field(default=5, gt=0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.1, gt=0.0, lt=1.0)
    model_config = SettingsConfigDict(env_prefix="DATA_")

class Settings(BaseSettings):
    models_path: Path = Field(default=Path("models"))
    logs_path: Path = Field(default=Path("logs"))
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()