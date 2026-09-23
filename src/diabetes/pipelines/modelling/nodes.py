"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 1.5.0
"""

import logging
from importlib import import_module
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

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


def optimize_hyperparameters(
    master_table: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """06_models: procura os melhores hiperparâmetros com GridSearchCV.

    Igual ao notebook: uma grade de hiperparâmetros (parameters_modelling.yml,
    bloco `param_grid`) é testada com validação cruzada só nas linhas de treino
    (`train_splits`). Como a validação é feita dentro do treino, não precisamos
    de um split de validação separado.

    Entradas:
        master_table: tabela mestre, com a coluna `split`.
        params: bloco `modelling_optimized`/`modelling_production` do
        parameters_modelling.yml (`class_path`, `init_args`, `param_grid`,
        `cv`, `scoring`, `train_splits`, `eval_splits`).

    Saída:
        Mesma estrutura de `train_model`, mas com o melhor estimador
        encontrado e os campos extras `best_params` e `best_score`.
    """

    target = params["target_column"]

    df_train = master_table[master_table["split"].isin(params["train_splits"])]

    feature_cols = [col for col in master_table.columns if col not in (target, "split")]

    X_train = df_train[feature_cols]
    y_train = df_train[target]

    cls = _load_class(params["class_path"])

    grid = GridSearchCV(
        cls(**params.get("init_args", {})),
        param_grid=params["param_grid"],
        cv=params.get("cv", 5),
        scoring=params.get("scoring", "roc_auc"),
        n_jobs=params.get("n_jobs", -1),
    )
    grid.fit(X_train, y_train)

    logger.info(
        f"Optimized {cls.__name__} on {len(X_train)} rows: "
        f"best {params.get('scoring', 'roc_auc')}={grid.best_score_:.4f} "
        f"with {grid.best_params_}"
    )

    return {
        "estimator": grid.best_estimator_,
        "target_column": target,
        "feature_columns": feature_cols,
        "eval_splits": params.get("eval_splits", []),
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
    }


def evaluate_model(
    model: dict[str, Any],
    master_table: pd.DataFrame,
) -> dict[str, Any]:
    """08_reporting: mede o desempenho do modelo em cada divisão (`eval_splits`).

    Para cada divisão listada em `eval_splits` (ex.: train e test), calcula as
    mesmas métricas do notebook: accuracy, precision, recall, F1 e ROC AUC.
    A mesma função serve para o modelo de referência, para o modelo otimizado
    e para o modelo de produção.

    Entradas:
        model: dicionário devolvido por `train_model`/`optimize_hyperparameters`.
        master_table: tabela mestre, com a coluna `split`.

    Saída:
        Dicionário simples (guardado em JSON em 08_reporting) com o nome do
        modelo e as métricas por divisão.
    """

    estimator = model["estimator"]
    target = model["target_column"]
    feature_cols = model["feature_columns"]

    metrics: dict[str, dict[str, float]] = {}
    for split in model.get("eval_splits", []):
        df_eval = master_table[master_table["split"] == split]
        if df_eval.empty:
            continue

        y_true = df_eval[target]
        y_pred = estimator.predict(df_eval[feature_cols])
        y_score = (
            estimator.predict_proba(df_eval[feature_cols])[:, 1]
            if hasattr(estimator, "predict_proba")
            else y_pred
        )

        metrics[split] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "n_rows": int(len(df_eval)),
        }

    result = {
        "model_class": type(estimator).__name__,
        "metrics": metrics,
    }
    if "best_params" in model:
        result["best_params"] = model["best_params"]
        result["best_cv_score"] = model["best_score"]

    logger.info(f"Evaluated {result['model_class']}: {metrics}")

    return result
