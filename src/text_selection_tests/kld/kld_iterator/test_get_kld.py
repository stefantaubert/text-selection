import numpy as np
import pytest
from text_selection.kld.kld_iterator import get_kld


def test_get_kld__equal_values__returns_zero():
  dist = np.array([0.5, 0.5])
  target_dist = np.array([0.5, 0.5])
  result = get_kld(dist, target_dist)
  assert result == 0


def test_get_kld__dist_nan__returns_inf():
  dist = np.array([np.nan, np.nan])
  target_dist = np.array([0.5, 0.5])
  result = get_kld(dist, target_dist)
  assert result == np.inf


def test_get_kld__one_in_dist_nan__returns_inf():
  dist = np.array([1.0, np.nan])
  target_dist = np.array([0.5, 0.5])
  result = get_kld(dist, target_dist)
  assert result == np.inf


def test_get_kld__target_dist_nan__returns_inf():
  dist = np.array([0.5, 0.5])
  target_dist = np.array([np.nan, np.nan])
  result = get_kld(dist, target_dist)
  assert result == np.inf


def test_get_kld__one_in_target_dist_nan__returns_inf():
  dist = np.array([0.5, 0.5])
  target_dist = np.array([1.0, np.nan])
  result = get_kld(dist, target_dist)
  assert result == np.inf


def test_get_kld__dist_and_target_dist_nan__returns_inf():
  dist = np.array([np.nan, np.nan])
  target_dist = np.array([np.nan, np.nan])
  result = get_kld(dist, target_dist)
  assert result == np.inf


def test_get_kld__one_in_dist_and_target_dist_nan__returns_inf():
  dist = np.array([1.0, np.nan])
  target_dist = np.array([1.0, np.nan])
  result = get_kld(dist, target_dist)
  assert result == np.inf


def test_empty_input__returns_zero():
  dist = np.array([], dtype=np.float64)
  target_dist = np.array([], dtype=np.float64)
  result = get_kld(dist, target_dist)
  assert result == 0.0


def test_dist_zero__raises_assert():
  dist = np.array([0], dtype=np.float64)
  target_dist = np.array([1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kld(dist, target_dist)


def test_dist_negative__raises_assert():
  dist = np.array([-1.0], dtype=np.float64)
  target_dist = np.array([-1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kld(dist, target_dist)


def test_target_dist_zero__raises_assert():
  dist = np.array([1.0], dtype=np.float64)
  target_dist = np.array([0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kld(dist, target_dist)


def test_target_dist_negative__raises_assert():
  dist = np.array([1.0], dtype=np.float64)
  target_dist = np.array([-1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kld(dist, target_dist)


def test_dist_multi_dim_array__raises_assert():
  dist = np.array([[1.0]], dtype=np.float64)
  target_dist = np.array([1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kld(dist, target_dist)


def test_target_dist_multi_dim_array__raises_assert():
  dist = np.array([1.0], dtype=np.float64)
  target_dist = np.array([[1.0]], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kld(dist, target_dist)
