import mlflow
from waterflow.experiment import preprocess_data, create_model_tuned


def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    dataframe_train, target_train, dataframe_test, target_test = preprocess_data()
    create_model_tuned(dataframe_train, target_train, dataframe_test, target_test)

if __name__ == "__main__":
    main()
