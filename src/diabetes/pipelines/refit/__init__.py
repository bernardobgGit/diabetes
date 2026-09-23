"""
Pipeline 'refit': reajusta os artefatos (imputer, limites de outliers,
encoder, scaler) e o classificador usando TODAS as linhas do dataset de
modelagem, gerando os artefatos de produção usados pela inferência.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
