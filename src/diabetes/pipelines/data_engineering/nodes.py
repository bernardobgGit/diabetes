"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.5.0
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

SENIOR_AGE = 50

BMI_UNDERWEIGHT_MAX, BMI_HEALTHY_MAX, BMI_OVERWEIGHT_MAX = 18.5, 25, 30
GLUCOSE_LOW_MAX, GLUCOSE_NORMAL_MAX, GLUCOSE_HIDDEN_MAX = 70, 100, 125

BINARY_UNIQUE_VALUES = 2


def clean_data(
    raw_diabetes_dataset_modelling: pd.DataFrame,
    columns: dict[str, Any],
) -> pd.DataFrame:
    """01_raw -> 02_intermediate: seleciona apenas as colunas.

    Mantém somente as colunas listadas em `columns` no parameters.yml (as
    variáveis e a variável alvo). Nada mais é alterado nesta etapa: não há
    tratamento de valores ausentes, de outliers, nem novas variáveis ou
    codificação.

    Entradas:
        raw_diabetes_dataset_modelling: dataset original (01_raw).
        columns: dicionário do parameters.yml com `features` e `target`.

    Saída:
        Cópia do dataset apenas com as colunas selecionadas (02_intermediate).
    """

    all_columns = [
        item
        for v in columns.values()
        for item in (v if isinstance(v, list) else [v])
        if item in raw_diabetes_dataset_modelling.columns
    ]

    df_flt = raw_diabetes_dataset_modelling[all_columns].copy()

    logger.info(f"Cleaned data: {df_flt.shape[0]} rows, {df_flt.shape[1]} columns")

    return df_flt


def mark_zeros_as_missing(
    df: pd.DataFrame,
    zero_columns: list[str],
) -> pd.DataFrame:
    """02_intermediate -> 03_primary: transforma os zeros em valores ausentes.

    Em colunas como Glucose, BloodPressure, SkinThickness, Insulin e BMI, o
    valor 0 significa "não medido" (um valor 0 seria biologicamente
    impossível). Aqui esses zeros passam a ser NaN, para serem preenchidos mais
    tarde, na etapa 04_feature.

    Entradas:
        df: dataset da camada 02_intermediate.
        zero_columns: colunas em que 0 significa "não medido" (parameters.yml).

    Saída:
        Cópia do dataset com NaN no lugar dos zeros (03_primary).
    """

    df = df.copy()
    zeros = (df[zero_columns] == 0).sum()
    df[zero_columns] = df[zero_columns].replace(0, np.nan)

    logger.info(
        f"Marked zeros as missing: {int(zeros.sum())} values in {len(zero_columns)} "
        f"columns {zeros.astype(int).to_dict()}"
    )

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """03_primary: divide as linhas em treino e teste.

    A divisão acontece ANTES de qualquer etapa que aprenda com os dados
    (imputação, limites de outliers, codificação e escala). Assim, as linhas de
    teste nunca influenciam o que é aprendido no treino (sem "data leakage").

    Entradas:
        df: dataset da camada 03_primary.
        test_size: proporção de linhas reservadas para teste (parameters.yml).
        random_state: semente do sorteio, para o resultado ser reproduzível.

    Saídas:
        Dois datasets: as linhas de treino e as linhas de teste.
    """

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    logger.info(f"Split data: {len(train)} train rows, {len(test)} test rows")

    return train, test


def fit_imputer(
    train: pd.DataFrame,
    columns: list[str],
    n_neighbors: int,
) -> dict[str, Any]:
    """04_feature: APRENDE a preencher os valores ausentes (só com o treino).

    Ajusta (fit) um RobustScaler e um KNNImputer nas linhas de treino. As
    colunas são escaladas primeiro, para que nenhuma domine o cálculo das
    distâncias entre linhas parecidas. Este nó não altera nenhum dataset:
    devolve a "receita" (scaler + imputer), guardada em 06_models e usada depois
    por `apply_imputer` nas linhas de treino e de teste (e, mais tarde, nos
    dados de inferência).

    Entradas:
        train: linhas de treino (03_primary).
        columns: colunas a preencher.
        n_neighbors: número de vizinhos (k) usados pelo KNN.

    Saída:
        Dicionário com as colunas, o scaler e o imputer já ajustados.
    """

    scaler = RobustScaler()
    scaled = scaler.fit_transform(train[columns])
    imputer = KNNImputer(n_neighbors=n_neighbors).fit(scaled)

    logger.info(
        f"Fitted the KNN imputer on {len(train)} train rows "
        f"(k={n_neighbors}, columns: {columns})"
    )

    return {"columns": columns, "scaler": scaler, "imputer": imputer}


def apply_imputer(
    df: pd.DataFrame,
    fitted_imputer: dict[str, Any],
) -> pd.DataFrame:
    """04_feature: PREENCHE os valores ausentes com a receita aprendida no treino.

    Usa o scaler e o imputer de `fit_imputer`, sem reajustá-los. Cada valor
    ausente é substituído pela média dos k vizinhos mais parecidos QUE ESTÃO NO
    TREINO. A mesma função é usada nas linhas de treino e nas de teste.

    Entradas:
        df: linhas de treino ou de teste.
        fitted_imputer: receita devolvida por `fit_imputer`.

    Saída:
        Cópia do dataset sem valores ausentes nas colunas imputadas.
    """

    df = df.copy()
    columns = fitted_imputer["columns"]
    scaler = fitted_imputer["scaler"]

    n_missing = int(df[columns].isnull().sum().sum())

    scaled = scaler.transform(df[columns])
    imputed = fitted_imputer["imputer"].transform(scaled)
    df[columns] = scaler.inverse_transform(imputed)

    logger.info(f"Imputed {n_missing} missing values in {len(df)} rows")

    return df


def fit_outlier_thresholds(
    train: pd.DataFrame,
    columns: list[str],
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, dict[str, float]]:
    """04_feature: APRENDE os limites de outliers de cada coluna (só com o treino).

    Para cada coluna, calcula nas linhas de treino dois quantis (parameters.yml,
    bloco `outliers`) e alarga o intervalo entre eles em 1,5 vezes:

        limite inferior = q_baixo - 1,5 * (q_alto - q_baixo)
        limite superior = q_alto  + 1,5 * (q_alto - q_baixo)

    Entradas:
        train: linhas de treino, já sem valores ausentes.
        columns: colunas a analisar.
        lower_quantile: quantil baixo (ex.: 0,05).
        upper_quantile: quantil alto (ex.: 0,95).

    Saída:
        Dicionário {coluna: {"lower": ..., "upper": ...}}, guardado em 06_models.
    """

    thresholds = {}
    for col in columns:
        q1 = train[col].quantile(lower_quantile)
        q3 = train[col].quantile(upper_quantile)
        iqr = q3 - q1
        thresholds[col] = {
            "lower": float(q1 - 1.5 * iqr),
            "upper": float(q3 + 1.5 * iqr),
        }

    logger.info(
        f"Worked out outlier limits for {len(columns)} columns on {len(train)} "
        f"train rows (quantiles {lower_quantile} and {upper_quantile})"
    )

    return thresholds


def apply_outlier_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """04_feature: LIMITA os valores extremos aos limites aprendidos no treino.

    Um valor abaixo do limite inferior passa a ser o limite inferior, e um valor
    acima do limite superior passa a ser o limite superior. Nenhuma linha é
    removida. A mesma função é usada nas linhas de treino e nas de teste.

    Entradas:
        df: linhas de treino ou de teste.
        thresholds: limites devolvidos por `fit_outlier_thresholds`.

    Saída:
        Cópia do dataset com os valores extremos limitados.
    """

    df = df.copy()
    n_capped = 0
    for col, limits in thresholds.items():
        outside = (df[col] < limits["lower"]) | (df[col] > limits["upper"])
        n_capped += int(outside.sum())
        df[col] = df[col].clip(lower=limits["lower"], upper=limits["upper"])

    logger.info(
        f"Capped {n_capped} outlier values across {len(thresholds)} columns "
        f"in {len(df)} rows"
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """04_feature: cria as novas variáveis (NEW_*) do notebook.

    Não aprende nada com os dados: as regras são fixas (cortes de idade, BMI,
    glicose e insulina), por isso a mesma função serve para treino e teste.

    Variáveis criadas:
        NEW_AGE_CAT: faixa etária (mature ou senior).
        NEW_BMI: categoria de BMI (Underweight, Healthy, Overweight, Obese).
        NEW_GLUCOSE: categoria de glicose (Normal, Prediabetes, Diabetes).
        NEW_AGE_BMI_NOM: BMI e faixa etária combinados.
        NEW_AGE_GLUCOSE_NOM: glicose e faixa etária combinadas.
        NEW_INSULIN_SCORE: insulina Normal ou Abnormal.
        NEW_GLUCOSE * INSULIN e NEW_GLUCOSE * PREGNANCIES: produtos de duas
        colunas.

    Entradas:
        df: linhas de treino ou de teste, já sem valores ausentes.

    Saída:
        Cópia do dataset com as novas colunas.
    """

    df = df.copy()
    n_columns_before = df.shape[1]

    # The dataset only has adults (Age >= 21), so two age groups are enough.
    age_group = np.where(df["Age"] >= SENIOR_AGE, "senior", "mature")
    bmi_group = np.select(
        [
            df["BMI"] < BMI_UNDERWEIGHT_MAX,
            df["BMI"] < BMI_HEALTHY_MAX,
            df["BMI"] < BMI_OVERWEIGHT_MAX,
        ],
        ["underweight", "healthy", "overweight"],
        default="obese",
    )
    glucose_group = np.select(
        [
            df["Glucose"] < GLUCOSE_LOW_MAX,
            df["Glucose"] < GLUCOSE_NORMAL_MAX,
            df["Glucose"] <= GLUCOSE_HIDDEN_MAX,
        ],
        ["low", "normal", "hidden"],
        default="high",
    )

    df["NEW_AGE_CAT"] = age_group
    df["NEW_BMI"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=["Underweight", "Healthy", "Overweight", "Obese"],
    )
    df["NEW_GLUCOSE"] = pd.cut(
        df["Glucose"],
        bins=[0, 140, 200, 300],
        labels=["Normal", "Prediabetes", "Diabetes"],
    )
    df["NEW_AGE_BMI_NOM"] = np.char.add(bmi_group, age_group)
    df["NEW_AGE_GLUCOSE_NOM"] = np.char.add(glucose_group, age_group)
    df["NEW_INSULIN_SCORE"] = np.where(
        df["Insulin"].between(16, 166), "Normal", "Abnormal"
    )
    df["NEW_GLUCOSE * INSULIN"] = df["Glucose"] * df["Insulin"]
    df["NEW_GLUCOSE * PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

    logger.info(
        f"Created {df.shape[1] - n_columns_before} new features for {len(df)} rows "
        f"({n_columns_before} -> {df.shape[1]} columns)"
    )

    return df


def _levels(column: pd.Series) -> list[str]:
    """The distinct values of a text or category column, sorted alphabetically.

    The declared order of a category is ignored on purpose: a column saved to
    CSV comes back as plain text, and the result must not depend on that.
    """

    return sorted(str(v) for v in column.dropna().unique())


def _warn_if_unseen(col: str, column: pd.Series, known: list[str]) -> None:
    unseen = set(column.dropna().astype(str).unique()) - set(known)
    if unseen:
        logger.warning(f"Column {col} has values not seen in train: {sorted(unseen)}")


def fit_encoder(train: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """04_feature -> 05_model_input: APRENDE como codificar as colunas de texto.

    Só com as linhas de treino, deteta automaticamente as colunas de texto ou
    de categoria e decide como codificar cada uma:

    - Coluna com 2 valores: vira uma única coluna 0/1 (`binary`: valor -> código).
    - Coluna com mais valores: vira uma coluna True/False por valor, exceto o
      primeiro, que serve de referência (`one_hot`: todos os valores, por ordem
      alfabética).

    O resultado é um dicionário simples, guardado em JSON (06_models), que uma
    pessoa consegue abrir e ler.

    Entradas:
        train: linhas de treino.

    Saída:
        Dicionário com as chaves `binary` e `one_hot`.
    """

    text_cols = train.select_dtypes(include=["object", "string", "category"]).columns

    binary, one_hot = {}, {}
    for col in text_cols:
        levels = _levels(train[col])
        if len(levels) == BINARY_UNIQUE_VALUES:
            binary[col] = {level: code for code, level in enumerate(levels)}
        else:
            one_hot[col] = levels

    logger.info(
        f"Fitted the encoder on {len(train)} train rows: "
        f"0/1 columns {list(binary)}, one-hot columns {list(one_hot)}"
    )

    return {"binary": binary, "one_hot": one_hot}


def apply_encoder(
    df: pd.DataFrame,
    encoder: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """04_feature -> 05_model_input: CODIFICA com a receita aprendida no treino.

    Como a receita é fixa, treino e teste ficam sempre com as mesmas colunas, na
    mesma ordem, mesmo que falte um valor (ou apareça um valor novo) no teste.
    Um valor nunca visto no treino fica com todas as colunas dummy a False e
    gera um aviso (WARNING) no log.

    Entradas:
        df: linhas de treino ou de teste.
        encoder: receita devolvida por `fit_encoder`.

    Saída:
        Cópia do dataset apenas com colunas numéricas ou True/False.
    """

    df = df.copy()
    n_columns_before = df.shape[1]

    for col, codes in encoder["binary"].items():
        _warn_if_unseen(col, df[col], list(codes))
        df[col] = df[col].astype("object").map(codes)

    dummies = {}
    for col, levels in encoder["one_hot"].items():
        _warn_if_unseen(col, df[col], levels)
        values = df[col].astype("object")
        for level in levels[1:]:
            dummies[f"{col}_{level}"] = values == level

    df = df.drop(columns=list(encoder["one_hot"]))
    encoded = pd.concat([df, pd.DataFrame(dummies, index=df.index)], axis=1)

    logger.info(
        f"Encoded {len(encoded)} rows ({n_columns_before} -> {encoded.shape[1]} columns)"
    )

    return encoded


def fit_scaler(train: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """05_model_input: APRENDE a escala de cada coluna (só com o treino).

    Ajusta um RobustScaler: guarda a mediana e o IQR (intervalo entre o 1.º e o
    3.º quartil) de cada coluna, calculados nas linhas de treino. Este nó não
    altera nenhum dataset: devolve a "receita", guardada em 06_models.

    Entradas:
        train: linhas de treino, já codificadas.
        columns: colunas a escalar (`scale_columns` no parameters.yml).

    Saída:
        Dicionário com as colunas e o scaler já ajustado.
    """

    scaler = RobustScaler().fit(train[columns])

    logger.info(f"Fitted the robust scaler on {len(train)} train rows: {columns}")

    return {"columns": columns, "scaler": scaler}


def apply_scaler(df: pd.DataFrame, fitted_scaler: dict[str, Any]) -> pd.DataFrame:
    """05_model_input: ESCALA as colunas com a receita aprendida no treino.

    Cada valor passa a ser (valor - mediana do treino) / IQR do treino. Os
    números usados vêm sempre do treino, mesmo quando as linhas são de teste.

    Entradas:
        df: linhas de treino ou de teste, já codificadas.
        fitted_scaler: receita devolvida por `fit_scaler`.

    Saída:
        Cópia do dataset com as colunas escaladas (05_model_input).
    """

    df = df.copy()
    columns = fitted_scaler["columns"]
    df[columns] = fitted_scaler["scaler"].transform(df[columns])

    logger.info(f"Scaled {len(columns)} columns in {len(df)} rows")

    return df
