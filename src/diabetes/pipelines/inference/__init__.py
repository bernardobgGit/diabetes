"""
Pipeline 'inference': aplica os artefatos de produção (gerados pelo refit)
nos dados novos e devolve as predições do classificador produtivo.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
