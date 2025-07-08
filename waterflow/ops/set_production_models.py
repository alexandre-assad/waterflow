import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion

client = MlflowClient()
MODELS_NAME = ["Waterflow XGBoost", "Waterflow Scaler"]


def get_staging_model(model_name: str) -> ModelVersion:
    return client.get_model_version_by_alias(name=model_name, alias="Staging")


def set_in_production(model_name: str, model: ModelVersion) -> None:
    try:
        production_model = client.get_model_version_by_alias(
            name=model_name, alias="Production"
        )
    except:
        production_model = None
    if production_model:
        client.set_registered_model_alias(
            production_model.name, "Archived", production_model.version
        )
        client.delete_registered_model_alias(production_model.name, "Production")

    client.set_registered_model_alias(model.name, "Production", model.version)
    client.delete_registered_model_alias(model.name, "Staging")


def main():
    for model_name in MODELS_NAME:
        try:
            staging_model = get_staging_model(model_name)
        except:
            continue
        if not staging_model:
            continue

        set_in_production(model_name, staging_model)
        # remove_all_staging(model_name)


if __name__ == "__main__":
    main()
