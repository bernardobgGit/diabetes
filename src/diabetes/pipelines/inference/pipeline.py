"""
Pipeline 'inference': encode & scale the inference data with the production
artifacts and predict with the productive classifier.
"""

from kedro.pipeline import Node, Pipeline

from diabetes.pipelines.data_engineering.nodes import (
    apply_encoder,
    apply_imputer,
    apply_outlier_thresholds,
    apply_scaler,
    clean_data,
    create_features,
    mark_zeros_as_missing,
)

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # As mesmas funções de transformação do data_engineering, mas sem
            # nenhum "fit": tudo usa os artefatos de produção do refit.
            Node(
                func=clean_data,
                inputs=["raw_diabetes_dataset_inference", "params:columns"],
                outputs="intermediate_diabetes_dataset_inference",
                name="inference_clean_data_node",
            ),
            Node(
                func=mark_zeros_as_missing,
                inputs=[
                    "intermediate_diabetes_dataset_inference",
                    "params:zero_columns",
                ],
                outputs="primary_diabetes_dataset_inference",
                name="inference_mark_zeros_as_missing_node",
            ),
            Node(
                func=apply_imputer,
                inputs=[
                    "primary_diabetes_dataset_inference",
                    "production_imputer_diabetes",
                ],
                outputs="imputed_diabetes_dataset_inference",
                name="inference_apply_imputer_node",
            ),
            Node(
                func=apply_outlier_thresholds,
                inputs=[
                    "imputed_diabetes_dataset_inference",
                    "production_outlier_thresholds_diabetes",
                ],
                outputs="capped_diabetes_dataset_inference",
                name="inference_apply_outlier_thresholds_node",
            ),
            Node(
                func=create_features,
                inputs="capped_diabetes_dataset_inference",
                outputs="feature_diabetes_dataset_inference",
                name="inference_create_features_node",
            ),
            Node(
                func=apply_encoder,
                inputs=[
                    "feature_diabetes_dataset_inference",
                    "production_encoder_diabetes",
                ],
                outputs="encoded_diabetes_dataset_inference",
                name="inference_apply_encoder_node",
            ),
            Node(
                func=apply_scaler,
                inputs=[
                    "encoded_diabetes_dataset_inference",
                    "production_scaler_diabetes",
                ],
                outputs="model_input_diabetes_dataset_inference",
                name="inference_apply_scaler_node",
            ),
            Node(
                func=predict,
                inputs=[
                    "model_input_diabetes_dataset_inference",
                    "production_model_diabetes",
                ],
                outputs="inference_predictions",
                name="inference_predict_node",
            ),
        ]
    )
