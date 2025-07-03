import mlflow

from typing import Any

from numpy import percentile, ndarray
from pandas import read_csv, DataFrame, concat

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

import mlflow

def to_register_model(model_name: str, score: float | None = None) -> bool:
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(name=model_name)
        if not versions:
            return True
        best_f1 = -1.0
        for version in versions:
            run_id = version._run_id
            if not run_id:
                continue
            run = client.get_run(run_id)
            metrics = run.data.metrics
            f1 = metrics.get("f1_score", -1)
            if f1 > best_f1:
                best_f1 = f1
        if score is None:
            return False
        return score > best_f1
    except Exception:
        return True


def balance_dataframe(dataframe: DataFrame) -> DataFrame:
    potable_class = dataframe[dataframe["Potability"] == 1]
    unpotablue_class = dataframe[dataframe["Potability"] == 0]
    unpotablue_class_downsample = resample(
        unpotablue_class, replace=False, n_samples=len(potable_class), random_state=42
    )
    df_balanced = concat([potable_class, unpotablue_class_downsample])
    return df_balanced


def drop_outliers(
    dataframe: DataFrame, columns: list[str], percent: int = 75
) -> DataFrame:
    """
    With Tukey's method to find outliers, we change quartile by input percentile & drop outliers
    """
    for column in columns:
        if dataframe[column].dtype not in ["int64", "float64"]:
            continue
        first_quartile = percentile(dataframe[column], 100 - percent)
        third_quartile = percentile(dataframe[column], percent)
        step = 1.5 * (third_quartile - first_quartile)

        dataframe.drop(
            dataframe.loc[
                ~(
                    (dataframe[column] >= first_quartile - step)
                    & (dataframe[column] <= third_quartile + step)
                ),
                column,
            ].index,
            inplace=True,
        )
    return dataframe


def preprocess_data(registry: bool = True) -> tuple[ndarray, Any, ndarray, Any]:
    # client = mlflow.tracking.MlflowClient()
    dataframe = read_csv("./data/water_potability.csv")
    dataframe_fill_na = dataframe.dropna()
    dataframe_sized = drop_outliers(dataframe_fill_na, dataframe.columns)
    dataframe_balanced = balance_dataframe(dataframe_sized)
    dataframe_clean, target = (
        dataframe_balanced.drop(columns=["Potability"]),
        dataframe_balanced["Potability"],
    )
    dataframe_train, dataframe_test, target_train, target_test = train_test_split(
        dataframe_clean, target, test_size=0.33, random_state=6
    )
    scaler = StandardScaler()
    dataframe_train_standard = scaler.fit_transform(dataframe_train, target_train)
    dataframe_test_standard = scaler.fit_transform(dataframe_test, target_train)
    if registry:
        if to_register_model("Waterflow Scaler"):
            mlflow.sklearn.log_model(
                scaler, "Scaler", registered_model_name="Waterflow Scaler"
            )
            # client.transition_model_version_stage(
            #     name="Waterflow Scaler",
            #     version='latest',
            #     stage="Staging",
            #     archive_existing_versions=False
            # )
        else:
            mlflow.sklearn.log_model(scaler, "Scaler")
    return dataframe_train_standard, target_train, dataframe_test_standard, target_test


def create_model(
    dataframe_train, target_train, dataframe_test, target_test, registry: bool = True
):
    xgboost = XGBClassifier()
    xgboost.fit(dataframe_train, target_train)
    predictions = xgboost.predict(dataframe_test)
    f1 = f1_score(target_test, predictions)
    if registry:
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(xgboost, "XGboost")


def create_model_tuned(
    dataframe_train, target_train, dataframe_test, target_test, registry: bool = True
):
    client = mlflow.tracking.MlflowClient()
    xgboost = XGBClassifier(
        objective="binary:logistic",
        nthread=4,
        learning_rate=0.3,
        max_depth=12,
        n_estimators=400,
    )
    xgboost.fit(dataframe_train, target_train)
    predictions = xgboost.predict(dataframe_test)
    f1 = f1_score(target_test, predictions)
    if registry:
        mlflow.log_param("objective", "binary:logistic")
        mlflow.log_param("nthread", 4)
        mlflow.log_param("learning_rate", 0.3)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("n_estimators", 400)
        mlflow.log_metric("f1_score", f1)
        if to_register_model("Waterflow XGBoost", f1):
            mlflow.sklearn.log_model(
                xgboost, "XGboost Tuned", registered_model_name="Waterflow XGBoost"
            )
            # client.transition_model_version_stage(
            #     name="Waterflow XGBoost",
            #     version='latest',
            #     stage="Staging",
            #     archive_existing_versions=False
            # )
        else:
            mlflow.sklearn.log_model(xgboost, "XGboost Tuned")


def main() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.set_experiment(experiment_name="experiment_water_quality")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Scaler"):
        dataframe_train, target_train, dataframe_test, target_test = preprocess_data()
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="XGBoost"):
        create_model(dataframe_train, target_train, dataframe_test, target_test)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name="XGBoost Tuned"
    ):
        create_model_tuned(dataframe_train, target_train, dataframe_test, target_test)
    mlflow.end_run()


if __name__ == "__main__":
    main()
