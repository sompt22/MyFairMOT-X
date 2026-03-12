"""Unit tests for KalmanFilter used in tracking."""
import pytest
np = pytest.importorskip("numpy")
from lib.tracking_utils.kalman_filter import KalmanFilter


@pytest.fixture
def kf():
    return KalmanFilter()


def test_initiate_returns_correct_shape(kf):
    measurement = np.array([100.0, 200.0, 1.0, 50.0])
    mean, covariance = kf.initiate(measurement)
    assert mean.shape == (8,)
    assert covariance.shape == (8, 8)


def test_initiate_mean_matches_measurement(kf):
    measurement = np.array([100.0, 200.0, 1.0, 50.0])
    mean, _ = kf.initiate(measurement)
    np.testing.assert_array_equal(mean[:4], measurement)
    np.testing.assert_array_equal(mean[4:], 0)


def test_predict_increases_uncertainty(kf):
    measurement = np.array([100.0, 200.0, 1.0, 50.0])
    mean, covariance = kf.initiate(measurement)
    _, covariance_pred = kf.predict(mean, covariance)
    # Predicted covariance should be >= initial (uncertainty grows)
    assert np.all(np.diag(covariance_pred) >= np.diag(covariance) - 1e-9)


def test_update_reduces_uncertainty(kf):
    measurement = np.array([100.0, 200.0, 1.0, 50.0])
    mean, covariance = kf.initiate(measurement)
    mean_pred, covariance_pred = kf.predict(mean, covariance)
    mean_upd, covariance_upd = kf.update(mean_pred, covariance_pred, measurement)
    # Updated covariance diagonal should be <= predicted
    assert np.all(np.diag(covariance_upd) <= np.diag(covariance_pred) + 1e-9)


def test_predict_state_shape(kf):
    measurement = np.array([50.0, 80.0, 0.8, 40.0])
    mean, cov = kf.initiate(measurement)
    mean_pred, cov_pred = kf.predict(mean, cov)
    assert mean_pred.shape == (8,)
    assert cov_pred.shape == (8, 8)


def test_gating_distance_shape(kf):
    measurement = np.array([100.0, 200.0, 1.0, 50.0])
    mean, covariance = kf.initiate(measurement)
    measurements = np.array([
        [100.0, 200.0, 1.0, 50.0],
        [110.0, 210.0, 1.1, 55.0],
        [500.0, 500.0, 2.0, 100.0],
    ])
    dist = kf.gating_distance(mean, covariance, measurements)
    assert dist.shape == (3,)
    assert dist[0] < dist[2]  # closer measurement has smaller distance
