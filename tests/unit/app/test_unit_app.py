import numpy as np
from src.waterflow.app import FEATURES, model

def test_model_is_loaded():
    assert hasattr(model, 'predict')

def test_predict_input_shape():
    sample_input = np.array([[7.0] * len(FEATURES)])
    result = model.predict(sample_input)
    assert len(result) == 1

def test_predict_output_type():
    sample_input = np.array([[7.0] * len(FEATURES)])
    result = model.predict(sample_input)[0]
    assert isinstance(result, (int, np.integer))