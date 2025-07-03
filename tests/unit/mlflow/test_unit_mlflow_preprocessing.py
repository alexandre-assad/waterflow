import pytest
import pandas as pd
import numpy as np
from waterflow.experiment import balance_dataframe, drop_outliers, preprocess_data

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'feature1': [1, 2, 1000, 3, 4],
        'feature2': [10, 20, 30, 40, 50],
        'Potability': [1, 1, 0, 0, 0]
    })

def test_balance_dataframe(sample_df):
    balanced = balance_dataframe(sample_df)
    count_potable = sum(balanced["Potability"] == 1)
    count_unpotable = sum(balanced["Potability"] == 0)
    assert count_potable == count_unpotable

def test_drop_outliers_removes_extreme_value(sample_df):
    cleaned = drop_outliers(sample_df.copy(), ["feature1"], percent=75)
    assert cleaned["feature1"].max() < 1000
    assert cleaned.shape[0] < sample_df.shape[0]

def test_preprocess_output_shapes():
    X_train, y_train, X_test, y_test = preprocess_data(registry=False)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]