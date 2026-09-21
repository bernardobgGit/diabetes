"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 1.5.0
"""

import logging
from importlib import import_module
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _load_class(class_path: str) -> type:
    """Devolve a classe indicada por um caminho de texto.

    Exemplo: "sklearn.linear_model.LogisticRegression" separa-se em módulo
    ("sklearn.linear_model") e nome da classe ("LogisticRegression"), importa o
    módulo e vai buscar a classe. Assim, o modelo é escolhido no parameters
    modelling.yml, sem mexer no código.
    """

    module_name, class_name = class_path.rsplit(".", 1)

    return getattr(import_module(module_name), class_name)


def build_master_table(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """05_model_input: junta treino e teste numa única tabela mestre.

    Acrescenta a coluna `split` ("train" ou "test"), que diz de qual das duas
    tabelas veio cada linha. Assim, os nós seguintes escolhem as linhas pelo nome
    da divisão (`train_splits` e `eval_splits` no parameters_modelling.yml).
    Nenhum valor é alterado: as linhas de teste continuam a ser só de teste.

    Entradas:
        train: linhas de treino, já codificadas e escaladas (05_model_input).
        test: linhas de teste, já codificadas e escaladas (05_model_input).

    Saída:
        Tabela com todas as linhas e a coluna extra `split`.
    """

    master_table = pd.concat(
        [train.assign(split="train"), test.assign(split="test")],
        ignore_index=True,
    )

    logger.info(
        f"Built master table: {len(master_table)} rows "
        f"({len(train)} train + {len(test)} test), {master_table.shape[1]} columns"
    )

    return master_table


def train_model(
    master_table: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """06_models: treina o modelo de referência (baseline) só com as linhas de treino.

    O modelo vem do parameters_modelling.yml (`class_path`), por isso trocar de
    LogisticRegression para outro modelo não exige mexer neste código. As
    variáveis (features) são todas as colunas, exceto a variável alvo e a coluna
    `split`. Os dados já chegam codificados e escalados, com o que foi aprendido
    só no treino.

    Entradas:
        master_table: tabela mestre, com a coluna `split`.
        params: bloco `modelling_baseline` do parameters_modelling.yml.

    Saída:
        Dicionário com o modelo treinado (`estimator`), a variável alvo, a lista
        de variáveis usadas e as divisões a avaliar (`eval_splits`), para o nó
        de avaliação usar mais tarde.
    """

    target = params["target_column"]

    df_train = master_table[master_table["split"].isin(params["train_splits"])]

    feature_cols = [col for col in master_table.columns if col not in (target, "split")]

    X_train = df_train[feature_cols]
    y_train = df_train[target]

    cls = _load_class(params["class_path"])

    estimator = cls(**params.get("init_args", {}))

    estimator.fit(X_train, y_train)

    logger.info(
        f"Trained {cls.__name__} on {len(X_train)} rows "
        f"(splits: {params['train_splits']}) with {len(feature_cols)} features"
    )

    return {
        "estimator": estimator,
        "target_column": target,
        "feature_columns": feature_cols,
        "eval_splits": params.get("eval_splits", []),
    }
