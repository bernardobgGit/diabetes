# diabetes

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

Kedro project built from the `notebooks/diabetes-prediction.ipynb` notebook
(Pima Indians Diabetes dataset) for the Deployment course. It is organised in
four pipelines:

- **data_engineering**: selects the columns, marks zeros as missing, splits
  train/test and then fits (train only) and applies the KNN imputer, the
  outlier thresholds, the feature engineering (`NEW_*` columns), the encoder
  and the robust scaler — no data leakage.
- **modelling**: builds the master table, trains the baseline model
  (LogisticRegression), optimises hyperparameters with GridSearchCV
  (RandomForestClassifier) and evaluates both on train/test, saving the
  metrics in `data/08_reporting`.
- **refit**: refits all the artifacts and the classifier on ALL the modelling
  rows, producing the production artifacts in `data/06_models`
  (`production_*`).
- **inference**: applies the production artifacts to
  `data/01_raw/diabetes-dataset-inference.csv` and predicts with the
  production model, writing `data/07_model_output/inference_predictions.csv`.

## How to run

Install dependencies with uv (the `uv.lock` pins every dependency):

```
uv sync
```

Run the full pipeline (data engineering + modelling + refit + inference):

```
uv run kedro run
```

Visualise the pipelines:

```
uv run kedro viz
```

## FastAPI

The project exposes the pipelines as a REST API (`src/diabetes/api.py`):

```
uv run uvicorn diabetes.api:app --host 0.0.0.0 --port 8000
```

Swagger docs at `http://localhost:8000/docs`. Endpoints:

- `GET /health` - liveness check
- `GET /datasets` and `GET /datasets/{name}` - exposes catalog datasets as
  JSON (e.g. `inference_predictions`, `raw_diabetes_dataset_modelling`)
- `POST /inference` - online prediction from JSON instances
- `POST /batch-inference` - runs inference on the catalog CSV
- `POST /train` + `GET /train/{run_id}` - re-trains asynchronously
  (data_engineering + modelling + refit)

## Docker

```
docker compose up --build
```

The API listens on `http://localhost:8000`. The container mounts `./data`
(read-write) so the artifacts produced by `POST /train` persist, and `./conf`
read-only.

## Project rules and guidelines

This Kedro project was generated using `kedro 1.5.0`. Take a look at the
[Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)
