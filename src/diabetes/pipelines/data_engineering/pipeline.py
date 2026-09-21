"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.5.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    apply_encoder,
    apply_imputer,
    apply_outlier_thresholds,
    apply_scaler,
    clean_data,
    create_features,
    fit_encoder,
    fit_imputer,
    fit_outlier_thresholds,
    fit_scaler,
    mark_zeros_as_missing,
    split_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # 01_raw -> 02_intermediate
            # Apenas filtramos as colunas que queremos do raw dataset.
            Node(
                func=clean_data,
                inputs=["raw_diabetes_dataset_modelling", "params:columns"],
                outputs="intermediate_diabetes_dataset_modelling",
                name="clean_data_node",
            ),
            # 02_intermediate -> 03_primary
            Node(
                func=mark_zeros_as_missing,
                inputs=[
                    "intermediate_diabetes_dataset_modelling",
                    "params:zero_columns",
                ],
                outputs="primary_diabetes_dataset_modelling",
                name="mark_zeros_as_missing_node",
            ),
            # Split BEFORE fitting anything, so the test rows never influence
            # the imputer or the outlier limits.
            Node(
                func=split_data,
                inputs=[
                    "primary_diabetes_dataset_modelling",
                    "params:split.test_size",
                    "params:split.random_state",
                ],
                outputs=[
                    "primary_diabetes_dataset_modelling_train",
                    "primary_diabetes_dataset_modelling_test",
                ],
                name="split_data_node",
            ),
            # 03_primary -> 04_feature: missing values
            # (fit on the train rows, then applied to both train and test)
            Node(
                func=fit_imputer,
                inputs=[
                    "primary_diabetes_dataset_modelling_train",
                    "params:zero_columns",
                    "params:knn_neighbors",
                ],
                outputs="imputer_diabetes_modelling",
                name="fit_imputer_node",
            ),
            Node(
                func=apply_imputer,
                inputs=[
                    "primary_diabetes_dataset_modelling_train",
                    "imputer_diabetes_modelling",
                ],
                outputs="imputed_diabetes_dataset_modelling_train",
                name="impute_train_node",
            ),
            Node(
                func=apply_imputer,
                inputs=[
                    "primary_diabetes_dataset_modelling_test",
                    "imputer_diabetes_modelling",
                ],
                outputs="imputed_diabetes_dataset_modelling_test",
                name="impute_test_node",
            ),
            # 04_feature: outliers
            Node(
                func=fit_outlier_thresholds,
                inputs=[
                    "imputed_diabetes_dataset_modelling_train",
                    "params:columns.features",
                    "params:outliers.lower_quantile",
                    "params:outliers.upper_quantile",
                ],
                outputs="outlier_thresholds_diabetes_modelling",
                name="fit_outlier_thresholds_node",
            ),
            Node(
                func=apply_outlier_thresholds,
                inputs=[
                    "imputed_diabetes_dataset_modelling_train",
                    "outlier_thresholds_diabetes_modelling",
                ],
                outputs="capped_diabetes_dataset_modelling_train",
                name="cap_outliers_train_node",
            ),
            Node(
                func=apply_outlier_thresholds,
                inputs=[
                    "imputed_diabetes_dataset_modelling_test",
                    "outlier_thresholds_diabetes_modelling",
                ],
                outputs="capped_diabetes_dataset_modelling_test",
                name="cap_outliers_test_node",
            ),
            # 04_feature: new features (no fitting, so the same function is used twice)
            Node(
                func=create_features,
                inputs="capped_diabetes_dataset_modelling_train",
                outputs="feature_diabetes_dataset_modelling_train",
                name="create_features_train_node",
            ),
            Node(
                func=create_features,
                inputs="capped_diabetes_dataset_modelling_test",
                outputs="feature_diabetes_dataset_modelling_test",
                name="create_features_test_node",
            ),
            # 04_feature -> 05_model_input: encoding
            # (the recipe is learned on the train rows, then applied to both)
            Node(
                func=fit_encoder,
                inputs="feature_diabetes_dataset_modelling_train",
                outputs="encoder_diabetes_modelling",
                name="fit_encoder_node",
            ),
            Node(
                func=apply_encoder,
                inputs=[
                    "feature_diabetes_dataset_modelling_train",
                    "encoder_diabetes_modelling",
                ],
                outputs="encoded_diabetes_dataset_modelling_train",
                name="encode_train_node",
            ),
            Node(
                func=apply_encoder,
                inputs=[
                    "feature_diabetes_dataset_modelling_test",
                    "encoder_diabetes_modelling",
                ],
                outputs="encoded_diabetes_dataset_modelling_test",
                name="encode_test_node",
            ),
            # 05_model_input: scaling
            Node(
                func=fit_scaler,
                inputs=[
                    "encoded_diabetes_dataset_modelling_train",
                    "params:scale_columns",
                ],
                outputs="scaler_diabetes_modelling",
                name="fit_scaler_node",
            ),
            Node(
                func=apply_scaler,
                inputs=[
                    "encoded_diabetes_dataset_modelling_train",
                    "scaler_diabetes_modelling",
                ],
                outputs="model_input_diabetes_dataset_modelling_train",
                name="scale_train_node",
            ),
            Node(
                func=apply_scaler,
                inputs=[
                    "encoded_diabetes_dataset_modelling_test",
                    "scaler_diabetes_modelling",
                ],
                outputs="model_input_diabetes_dataset_modelling_test",
                name="scale_test_node",
            ),
        ]
    )
