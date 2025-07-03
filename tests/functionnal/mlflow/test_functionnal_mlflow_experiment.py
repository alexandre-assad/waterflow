import mlflow
from pathlib import Path

from mlflow.tracking import MlflowClient
from waterflow.experiment import preprocess_data, create_model, create_model_tuned


def test_mlflow_tracking(tmp_path):
    mlflow.set_tracking_uri(Path(tmp_path).absolute().as_uri())
    mlflow.set_experiment("test_exp")
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("metric1", 0.9)


def test_full_pipeline_logging(tmp_path):
    mlflow.set_experiment("test_experiment")
    X_train, y_train, X_test, y_test = preprocess_data(registry=False)

    with mlflow.start_run(run_name="XGBoost"):
        create_model(X_train, y_train, X_test, y_test)
        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_model_tuned_logs_params():
    mlflow.set_experiment("test_experiment")
    X_train, y_train, X_test, y_test = preprocess_data(registry=False)

    with mlflow.start_run(run_name="XGBoost Tuned"):
        create_model_tuned(X_train, y_train, X_test, y_test)
        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    params = client.get_run(run_id).data.params
    metrics = client.get_run(run_id).data.metrics

    for param in ["learning_rate", "n_estimators", "max_depth"]:
        assert param in params
    assert metrics["f1_score"] > 0.63
