"""
Pipeline 'refit': fit encoders & scalers on ALL the modelling data and train
the productive classifier, as in the course architecture.
"""

from kedro.pipeline import Node, Pipeline

from diabetes.pipelines.data_engineering.nodes import (
    apply_encoder,
    apply_imputer,
    apply_outlier_thresholds,
    apply_scaler,
    create_features,
    fit_encoder,
    fit_imputer,
    fit_outlier_thresholds,
    fit_scaler,
    mark_zeros_as_missing,
)
from diabetes.pipelines.modelling.nodes import optimize_hyperparameters

from .nodes import add_split_column


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Mesma sequência de transformações do data_engineering, mas os
            # "fits" aprendem com TODAS as linhas (não só com o treino).
            Node(
                func=mark_zeros_as_missing,
                inputs=[
                    "intermediate_diabetes_dataset_modelling",
                    "params:zero_columns",
                ],
                outputs="refit_primary_diabetes_dataset",
                name="refit_mark_zeros_as_missing_node",
            ),
            Node(
                func=fit_imputer,
                inputs=[
                    "refit_primary_diabetes_dataset",
                    "params:zero_columns",
                    "params:knn_neighbors",
                ],
                outputs="production_imputer_diabetes",
                name="refit_fit_imputer_node",
            ),
            Node(
                func=apply_imputer,
                inputs=[
                    "refit_primary_diabetes_dataset",
                    "production_imputer_diabetes",
                ],
                outputs="refit_imputed_diabetes_dataset",
                name="refit_apply_imputer_node",
            ),
            Node(
                func=fit_outlier_thresholds,
                inputs=[
                    "refit_imputed_diabetes_dataset",
                    "params:columns.features",
                    "params:outliers.lower_quantile",
                    "params:outliers.upper_quantile",
                ],
                outputs="production_outlier_thresholds_diabetes",
                name="refit_fit_outlier_thresholds_node",
            ),
            Node(
                func=apply_outlier_thresholds,
                inputs=[
                    "refit_imputed_diabetes_dataset",
                    "production_outlier_thresholds_diabetes",
                ],
                outputs="refit_capped_diabetes_dataset",
                name="refit_apply_outlier_thresholds_node",
            ),
            Node(
                func=create_features,
                inputs="refit_capped_diabetes_dataset",
                outputs="refit_feature_diabetes_dataset",
                name="refit_create_features_node",
            ),
            Node(
                func=fit_encoder,
                inputs="refit_feature_diabetes_dataset",
                outputs="production_encoder_diabetes",
                name="refit_fit_encoder_node",
            ),
            Node(
                func=apply_encoder,
                inputs=[
                    "refit_feature_diabetes_dataset",
                    "production_encoder_diabetes",
                ],
                outputs="refit_encoded_diabetes_dataset",
                name="refit_apply_encoder_node",
            ),
            Node(
                func=fit_scaler,
                inputs=[
                    "refit_encoded_diabetes_dataset",
                    "params:scale_columns",
                ],
                outputs="production_scaler_diabetes",
                name="refit_fit_scaler_node",
            ),
            Node(
                func=apply_scaler,
                inputs=[
                    "refit_encoded_diabetes_dataset",
                    "production_scaler_diabetes",
                ],
                outputs="refit_model_input_diabetes_dataset",
                name="refit_apply_scaler_node",
            ),
            # Todas as linhas marcadas como 'train' para o fit do classificador
            # produtivo com os mesmos parâmetros do modelling.
            Node(
                func=add_split_column,
                inputs=[
                    "refit_model_input_diabetes_dataset",
                    "params:refit_split_name",
                ],
                outputs="refit_master_table_diabetes_dataset",
                name="refit_add_split_column_node",
            ),
            Node(
                func=optimize_hyperparameters,
                inputs=[
                    "refit_master_table_diabetes_dataset",
                    "params:modelling_production",
                ],
                outputs="production_model_diabetes",
                name="refit_train_production_model_node",
            ),
        ]
    )
