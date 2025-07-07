import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion

client = MlflowClient()
MODELS_NAME = ['Waterflow XGBoost', 'Waterflow Scaler']

def get_staging_models(model_name: str) -> Model:
    return client.get_model_version_by_alias(name=model_name, alias="staging")

def select_staging_model(models):
    ...

def set_in_production(model) -> None:
    ...

def main():
    for model_name in MODELS_NAME:
        staging_models = get_staging_models(model_name)
        if not staging_models:
            continue

        model = select_staging_model(staging_models)
        set_in_production(model)

if __name__ == '__main__':
    main()
