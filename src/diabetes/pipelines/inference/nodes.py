"""
Nodes of the 'inference' pipeline.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def predict(
    df: pd.DataFrame,
    model: dict[str, Any],
) -> pd.DataFrame:
    """07_model_output: gera as predições com o classificador produtivo.

    Nenhum ajuste (fit) acontece aqui: os dados já chegam transformados pelos
    artefatos de produção e o modelo apenas prevê. As colunas de entrada são
    mantidas na saída, junto com a classe prevista (`prediction`) e, quando o
    modelo suporta, a probabilidade da classe positiva (`probability`).

    Entradas:
        df: dados de inferência já codificados e escalados.
        model: dicionário devolvido por `train_model`/`optimize_hyperparameters`
        do pipeline de refit (`production_model_diabetes`).

    Saída:
        Cópia do dataset de entrada com as colunas `prediction` e
        `probability` (07_model_output).
    """

    estimator = model["estimator"]
    feature_cols = model["feature_columns"]

    X = df[feature_cols]

    predictions = df.copy()
    predictions["prediction"] = estimator.predict(X)
    if hasattr(estimator, "predict_proba"):
        predictions["probability"] = estimator.predict_proba(X)[:, 1]

    logger.info(
        f"Predicted {len(predictions)} rows with "
        f"{type(estimator).__name__}: "
        f"{int(predictions['prediction'].sum())} positive"
    )

    return predictions
