"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 1.5.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import build_master_table, train_model


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
        ]
    )
