"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 1.5.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_master_table,
    evaluate_model,
    optimize_hyperparameters,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # 05_model_input: treino e teste numa só tabela, com a coluna `split`
            Node(
                func=build_master_table,
                inputs=[
                    "model_input_diabetes_dataset_modelling_train",
                    "model_input_diabetes_dataset_modelling_test",
                ],
                outputs="master_table_diabetes_dataset_modelling",
                name="build_master_table_node",
            ),
            # 06_models: modelo de referência, treinado só com as linhas de treino
            Node(
                func=train_model,
                inputs=[
                    "master_table_diabetes_dataset_modelling",
                    "params:modelling_baseline",
                ],
                outputs="baseline_model_diabetes_modelling",
                name="train_baseline_model_node",
            ),
            # 08_reporting: métricas do modelo de referência (train e test)
            Node(
                func=evaluate_model,
                inputs=[
                    "baseline_model_diabetes_modelling",
                    "master_table_diabetes_dataset_modelling",
                ],
                outputs="baseline_metrics_diabetes_modelling",
                name="evaluate_baseline_model_node",
            ),
            # 06_models: otimização de hiperparâmetros com GridSearchCV
            # (a validação cruzada acontece dentro das linhas de treino)
            Node(
                func=optimize_hyperparameters,
                inputs=[
                    "master_table_diabetes_dataset_modelling",
                    "params:modelling_optimized",
                ],
                outputs="optimized_model_diabetes_modelling",
                name="optimize_hyperparameters_node",
            ),
            # 08_reporting: métricas do modelo otimizado (train e test)
            Node(
                func=evaluate_model,
                inputs=[
                    "optimized_model_diabetes_modelling",
                    "master_table_diabetes_dataset_modelling",
                ],
                outputs="optimized_metrics_diabetes_modelling",
                name="evaluate_optimized_model_node",
            ),
        ]
    )
