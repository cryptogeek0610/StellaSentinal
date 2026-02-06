"""Unit tests for SAAPAnomalyPredictor."""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from projects.SOTI_Advanced_Analytics_Plus.ai_service.anomaly_predictor import SAAPAnomalyPredictor


@pytest.fixture
def predictor():
    return SAAPAnomalyPredictor(contamination=0.1)


@pytest.fixture
def trained_predictor(predictor):
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    predictor.train(X)
    return predictor


class TestInit:
    def test_default_contamination(self):
        p = SAAPAnomalyPredictor()
        assert p.model.contamination == 0.05

    def test_custom_contamination(self):
        p = SAAPAnomalyPredictor(contamination=0.2)
        assert p.model.contamination == 0.2

    def test_not_fitted_initially(self):
        p = SAAPAnomalyPredictor()
        assert p._is_fitted is False


class TestTrain:
    def test_valid_data(self, predictor):
        X = np.random.randn(50, 4)
        predictor.train(X)
        assert predictor._is_fitted is True

    def test_empty_array_raises(self, predictor):
        with pytest.raises(ValueError, match="empty"):
            predictor.train(np.array([]))

    def test_nan_values_raises(self, predictor):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="NaN"):
            predictor.train(X)

    def test_inf_values_raises(self, predictor):
        X = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Inf"):
            predictor.train(X)

    def test_single_feature(self, predictor):
        X = np.random.randn(50, 1)
        predictor.train(X)
        assert predictor._is_fitted is True


class TestPredict:
    def test_predict_after_train(self, trained_predictor):
        X = np.random.randn(10, 3)
        preds, scores = trained_predictor.predict(X)
        assert preds.shape == (10,)
        assert scores.shape == (10,)
        assert set(preds).issubset({-1, 1})

    def test_predict_before_train(self, predictor):
        with pytest.raises(NotFittedError):
            predictor.predict(np.random.randn(5, 3))

    def test_predict_empty_data(self, trained_predictor):
        with pytest.raises(ValueError, match="empty"):
            trained_predictor.predict(np.array([]))


class TestModelPersistence:
    def test_save_and_load(self, trained_predictor, tmp_path):
        model_path = tmp_path / "model.joblib"
        trained_predictor.save_model(model_path)
        assert model_path.exists()

        new_predictor = SAAPAnomalyPredictor()
        new_predictor.load_model(model_path)
        assert new_predictor._is_fitted is True

        X = np.random.randn(5, 3)
        preds_orig, _ = trained_predictor.predict(X)
        preds_loaded, _ = new_predictor.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_save_unfitted_raises(self, predictor, tmp_path):
        with pytest.raises(NotFittedError):
            predictor.save_model(tmp_path / "model.joblib")

    def test_load_missing_file(self, predictor, tmp_path):
        with pytest.raises(FileNotFoundError):
            predictor.load_model(tmp_path / "nonexistent.joblib")
