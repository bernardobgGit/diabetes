"""
FastAPI layer: exposes the Kedro pipelines as REST endpoints.

- GET  /health                 - liveness check
- GET  /datasets               - lists the datasets exposed by the API
- GET  /datasets/{name}        - returns a catalog dataset as JSON records
- POST /inference              - online prediction for one patient (query params)
- POST /inference/instances    - online prediction from JSON instances
- POST /batch-inference        - runs the inference pipeline on the catalog CSV
- POST /train                  - runs data_engineering + modelling + refit
- GET  /train/{run_id}         - polls the status of a training run

Run with:
    uv run uvicorn diabetes.api:app --host 0.0.0.0 --port 8000
"""

import logging
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from kedro.framework.project import configure_project, pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_PATH = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "diabetes"

# Datasets que a API expõe para leitura (treinamento e resultados).
EXPOSED_DATASETS = {
    "raw_diabetes_dataset_modelling",
    "raw_diabetes_dataset_inference",
    "master_table_diabetes_dataset_modelling",
    "inference_predictions",
}

TRAIN_PIPELINES = ["data_engineering", "modelling", "refit"]

app = FastAPI(
    title="diabetes",
    description="Kedro diabetes pipelines exposed as a REST API.",
    version="0.1",
)

_state = {"bootstrapped": False}
_bootstrap_lock = threading.Lock()
_run_lock = threading.Lock()
_train_runs: dict[str, dict[str, Any]] = {}


class PatientFeatures(BaseModel):
    """Pydantic contract: the raw feature columns of ONE patient."""

    model_config = {"extra": "forbid"}

    Pregnancies: int = Field(
        ge=0, description="Number of times pregnant.", examples=[6]
    )
    Glucose: int = Field(
        ge=0,
        description="Plasma glucose concentration (2h oral glucose tolerance "
        "test), mg/dL.",
        examples=[148],
    )
    BloodPressure: int = Field(
        ge=0, description="Diastolic blood pressure, mm Hg.", examples=[72]
    )
    SkinThickness: int = Field(
        ge=0, description="Triceps skin fold thickness, mm.", examples=[35]
    )
    Insulin: int = Field(
        ge=0, description="2-hour serum insulin, mu U/ml.", examples=[0]
    )
    BMI: float = Field(
        ge=0, description="Body mass index, kg/m^2.", examples=[33.6]
    )
    DiabetesPedigreeFunction: float = Field(
        ge=0,
        description="Diabetes likelihood based on family history.",
        examples=[0.627],
    )
    Age: int = Field(ge=0, description="Age, years.", examples=[50])


class InferenceRequest(BaseModel):
    """Pydantic contract: a list of patients with the raw feature columns."""

    instances: list[PatientFeatures] = Field(
        min_length=1,
        examples=[
            [
                {
                    "Pregnancies": 6,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                },
                {
                    "Pregnancies": 1,
                    "Glucose": 85,
                    "BloodPressure": 66,
                    "SkinThickness": 29,
                    "Insulin": 0,
                    "BMI": 26.6,
                    "DiabetesPedigreeFunction": 0.351,
                    "Age": 31,
                },
            ]
        ],
    )


def _ensure_bootstrapped() -> None:
    """Bootstraps the Kedro project once (double-checked locking)."""

    if not _state["bootstrapped"]:
        with _bootstrap_lock:
            if not _state["bootstrapped"]:
                bootstrap_project(PROJECT_PATH)
                configure_project(PACKAGE_NAME)
                _state["bootstrapped"] = True


def _run_pipeline(
    pipeline_names: list[str],
    catalog_overrides: dict[str, MemoryDataset] | None = None,
) -> Any:
    """Runs pipeline(s) inside a KedroSession and returns the catalog.

    `catalog_overrides` lets the API inject request data into the catalog
    (MemoryDataset pattern), so the SAME pipeline used in batch mode also
    serves online inference.
    """

    _ensure_bootstrapped()
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = session.load_context().catalog

        pipeline = sum(pipelines[name] for name in pipeline_names)

        for name, dataset in (catalog_overrides or {}).items():
            catalog[name] = dataset

        SequentialRunner().run(pipeline, catalog)

    return catalog


def _train(run_id: str) -> None:
    try:
        with _run_lock:
            _run_pipeline(TRAIN_PIPELINES)
        _train_runs[run_id]["status"] = "success"
    except Exception as exc:  # noqa: BLE001 - report any failure to the poller
        logger.exception("Training run %s failed", run_id)
        _train_runs[run_id]["status"] = "failed"
        _train_runs[run_id]["error"] = str(exc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets() -> dict[str, list[str]]:
    return {"datasets": sorted(EXPOSED_DATASETS)}


@app.get("/datasets/{dataset_name}")
def get_dataset(dataset_name: str, limit: int | None = None) -> dict[str, Any]:
    """Exposes a catalog dataset as JSON records."""

    if dataset_name not in EXPOSED_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not exposed. "
            f"Available: {sorted(EXPOSED_DATASETS)}",
        )

    _ensure_bootstrapped()
    try:
        with KedroSession.create(project_path=PROJECT_PATH) as session:
            df: pd.DataFrame = session.load_context().catalog.load(dataset_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if limit is not None:
        df = df.head(limit)

    return {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "records": df.to_dict(orient="records"),
    }


def _predict(instances: list[PatientFeatures]) -> dict[str, Any]:
    """Runs the inference pipeline on the given patients.

    The request data is injected into the catalog as a MemoryDataset and the
    SAME inference pipeline used in batch mode is executed, so online and
    batch predictions are always consistent.
    """

    overrides = {
        "raw_diabetes_dataset_inference": MemoryDataset(
            data=pd.DataFrame([p.model_dump() for p in instances])
        ),
        "inference_predictions": MemoryDataset(),
    }

    try:
        with _run_lock:
            catalog = _run_pipeline(["inference"], catalog_overrides=overrides)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=409,
            detail=f"Inference failed (run POST /train first if the "
            f"production artifacts do not exist yet): {exc}",
        ) from exc

    predictions: pd.DataFrame = catalog.load("inference_predictions")
    output_cols = [c for c in ("prediction", "probability") if c in predictions]

    return {
        "n_predictions": int(len(predictions)),
        "predictions": predictions[output_cols].to_dict(orient="records"),
    }


@app.post("/inference")
def run_inference(
    features: Annotated[PatientFeatures, Query()],
) -> dict[str, Any]:
    """Online inference for ONE patient, with the production model.

    Each feature is a query parameter (listed under "Parameters" in Swagger).
    """

    return _predict([features])


@app.post("/inference/instances")
def run_inference_instances(request: InferenceRequest) -> dict[str, Any]:
    """Online inference for SEVERAL patients, sent as a JSON body."""

    return _predict(request.instances)


@app.post("/batch-inference")
def run_batch_inference() -> dict[str, Any]:
    """Batch inference: predicts the CSV in the catalog (data/01_raw)."""

    try:
        with _run_lock:
            catalog = _run_pipeline(["inference"])
    except Exception as exc:
        logger.exception("Batch inference failed")
        raise HTTPException(
            status_code=409,
            detail=f"Batch inference failed (run POST /train first if the "
            f"production artifacts do not exist yet): {exc}",
        ) from exc

    predictions: pd.DataFrame = catalog.load("inference_predictions")

    return {
        "n_predictions": int(len(predictions)),
        "dataset": "inference_predictions",
        "predictions": predictions[["prediction", "probability"]].to_dict(
            orient="records"
        ),
    }


@app.post("/train", status_code=202)
def train() -> dict[str, str]:
    """Async training: runs data_engineering + modelling + refit pipelines."""

    run_id = uuid.uuid4().hex[:12]
    _train_runs[run_id] = {"status": "running"}
    threading.Thread(target=_train, args=(run_id,), daemon=True).start()

    return {"run_id": run_id, "status": "running"}


@app.get("/train/{run_id}")
def train_status(run_id: str) -> dict[str, Any]:
    run = _train_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id '{run_id}'")

    return {"run_id": run_id, **run}
