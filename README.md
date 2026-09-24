# diabetes

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Sobre o projeto

Projeto Kedro construído a partir do notebook
`notebooks/diabetes-prediction.ipynb` (dataset Pima Indians Diabetes), para a
disciplina de Deployment. O trabalho do notebook foi reorganizado em quatro
pipelines:

- **data_engineering**: seleciona as colunas, marca os zeros como valores
  ausentes, separa treino/teste e então ajusta (só no treino) e aplica o
  imputador KNN, os limites de outliers, o feature engineering (colunas
  `NEW_*`), o encoder e o RobustScaler — sem data leakage.
- **modelling**: monta a master table, treina o modelo baseline
  (LogisticRegression), otimiza hiperparâmetros com GridSearchCV
  (RandomForestClassifier) e avalia os dois em treino/teste, salvando as
  métricas em `data/08_reporting`.
- **refit**: reajusta todos os artefatos e o classificador com TODAS as
  linhas do dataset, gerando os artefatos de produção em `data/06_models`
  (`production_*`).
- **inference**: aplica os artefatos de produção no dataset
  `data/01_raw/diabetes-dataset-inference.csv` e prevê com o modelo de
  produção, gerando `data/07_model_output/inference_predictions.csv`.

## Pré-requisitos

- Python 3.14+ — `python --version`
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes (`pip install uv`
  ou `winget install astral-sh.uv`)
- Docker Desktop (somente para a parte de container)

## 1. Instalação

Clone o repositório e instale as dependências com uv (o `uv.lock` fixa todas
as versões):

```bash
git clone https://github.com/bernardobgGit/diabetes.git
cd diabetes
uv sync
```

## 2. Executando os pipelines (Kedro)

Rodar tudo (data_engineering + modelling + refit + inference):

```bash
uv run kedro run
```

Rodar um pipeline por vez:

```bash
uv run kedro run --pipeline=data_engineering
uv run kedro run --pipeline=modelling
uv run kedro run --pipeline=refit
uv run kedro run --pipeline=inference
```

> Atenção: `refit` depende da saída de `data_engineering`, e `inference`
> depende dos artefatos de `refit`. Rode-os na ordem acima (ou `kedro run`
> completo, que resolve a ordem sozinho).

Visualizar o grafo dos pipelines (Kedro-Viz):

```bash
uv run kedro viz
# abre em http://localhost:4141
```

Saídas geradas: modelos e artefatos em `data/06_models`, predições em
`data/07_model_output/inference_predictions.csv`, métricas em
`data/08_reporting/*.json`.

## 3. Usando a API (FastAPI)

Suba o servidor:

```bash
uv run uvicorn diabetes.api:app --host 0.0.0.0 --port 8000
```

Documentação interativa (Swagger): http://localhost:8000/docs

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | verifica se a API está no ar |
| GET | `/datasets` | lista os datasets expostos |
| GET | `/datasets/{nome}` | devolve um dataset do catálogo em JSON |
| POST | `/inference` | predição online a partir de JSON |
| POST | `/batch-inference` | roda inferência no CSV do catálogo |
| POST | `/train` | retreina (data_eng + modelling + refit), assíncrono |
| GET | `/train/{run_id}` | consulta o status de um treino |

### Exemplos

Verificar se está no ar:

```bash
curl http://localhost:8000/health
```

Ler um dataset como JSON (o dataset `inference_predictions` contém as
predições da última inferência em lote):

```bash
curl http://localhost:8000/datasets
curl "http://localhost:8000/datasets/inference_predictions?limit=5"
curl "http://localhost:8000/datasets/raw_diabetes_dataset_modelling?limit=5"
```

Predição online (envia instâncias brutas, recebe `prediction` e
`probability`):

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d "{\"instances\": [{\"Pregnancies\": 6, \"Glucose\": 148, \"BloodPressure\": 72, \"SkinThickness\": 35, \"Insulin\": 0, \"BMI\": 33.6, \"DiabetesPedigreeFunction\": 0.627, \"Age\": 50}]}"
```

Resposta:

```json
{"n_predictions": 1, "predictions": [{"prediction": 1, "probability": 0.81}]}
```

Inferência em lote (usa o CSV `data/01_raw/diabetes-dataset-inference.csv` e
salva em `inference_predictions`):

```bash
curl -X POST http://localhost:8000/batch-inference
```

Retreinar tudo (necessário se os artefatos `data/06_models/production_*`
ainda não existirem):

```bash
curl -X POST http://localhost:8000/train
# {"run_id": "abc123...", "status": "running"}
curl http://localhost:8000/train/abc123...
# {"run_id": "abc123...", "status": "success"}
```

> Se `/inference` responder erro 409, os artefatos de produção ainda não
> existem: rode `uv run kedro run` ou `POST /train` primeiro.

## 4. Rodando com Docker

Pré-requisito: **Docker Desktop aberto** (no Windows, ele precisa estar
rodando antes dos comandos).

Na raiz do projeto:

```bash
docker compose up --build
```

Isso constrói a imagem (instala as dependências com `uv sync` a partir do
`uv.lock`) e sobe a API. Quando aparecer `Uvicorn running on
http://0.0.0.0:8000`, a API está disponível em:

- http://localhost:8000/docs — Swagger
- http://localhost:8000/health

Teste os mesmos endpoints da seção anterior (`/inference`,
`/batch-inference`, `/datasets/...`). O compose monta `./data` no container,
então os artefatos gerados por `POST /train` ficam salvos na pasta `data/`
do projeto.

Comandos úteis:

```bash
docker compose up -d      # sobe em segundo plano
docker compose down       # para e remove o container
docker compose up --build # reconstrói após mudar o código
docker logs -f repo-api-1 # acompanha os logs
```

## Estrutura do projeto

```
conf/base/            catalog.yml (datasets), parameters*.yml (config)
data/01_raw/          datasets de entrada (modelling e inference)
data/06_models/       modelos e artefatos de produção
data/07_model_output/ inference_predictions.csv
data/08_reporting/    métricas dos modelos (JSON)
src/diabetes/
  api.py              API FastAPI (endpoints)
  pipelines/
    data_engineering/ limpeza, split, imputer, outliers, features, encoding, scaler
    modelling/        master table, treino baseline, GridSearchCV, avaliação
    refit/            refit dos artefatos e do classificador em todos os dados
    inference/        transform + predict com os artefatos de produção
Dockerfile            imagem da API (uv + uvicorn)
docker-compose.yml    sobe a API na porta 8000 com conf/ e data/ montados
```

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
