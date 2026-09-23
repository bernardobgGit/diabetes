"""
Nodes of the 'refit' pipeline.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_split_column(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Marca todas as linhas com o mesmo nome de divisão.

    No refit não existe separação treino/teste: depois de validar a abordagem
    no pipeline de modelling, todas as linhas do dataset de modelagem são
    usadas para treinar os artefatos de produção. A coluna `split` é criada
    só para que `train_model`/`optimize_hyperparameters` (que filtram pelas
    divisões em `train_splits`) possam ser reutilizados sem alteração.

    Entradas:
        df: dataset já processado (05_model_input), com todas as linhas.
        split_name: valor da coluna `split` (parameters_refit.yml).

    Saída:
        Cópia do dataset com a coluna `split` preenchida.
    """

    df = df.assign(split=split_name)

    logger.info(f"Marked all {len(df)} rows as split '{split_name}'")

    return df
