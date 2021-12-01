import numpy as np
from text_selection.kld.kld_iterator import is_valid_distribution


def test_axis_0_empty_array__returns_true():
  qk = np.array([], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert result


def test_axis_1_empty_array__returns_true():
  qk = np.array([[]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_axis_0_non_empty__returns_true():
  qk = np.array([1.0], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert result


def test_axis_1_non_empty__returns_true():
  qk = np.array([[1.0]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_axis_0_sum__returns_true():
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert result


def test_axis_1_sum__returns_true():
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_axis_1_multi_sum__returns_true():
  qk = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_axis_0_bigger_than_one__returns_false():
  qk = np.array([1.1], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_1_bigger_than_one__returns_false():
  qk = np.array([[1.1]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_0_smaller_than_one__returns_false():
  qk = np.array([0.9], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_1_smaller_than_one__returns_false():
  qk = np.array([0.9], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_0_zero__returns_false():
  qk = np.array([0.0], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_1_zero__returns_false():
  qk = np.array([[0.0]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_axis_0_negative__returns_false():
  qk = np.array([-0.1], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_1_negative__returns_false():
  qk = np.array([[-0.1]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_axis_0_nan__returns_false():
  qk = np.array([np.nan], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_1_nan__returns_false():
  qk = np.array([[np.nan]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_axis_0_inf__returns_false():
  qk = np.array([np.inf], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_1_inf__returns_false():
  qk = np.array([[np.inf]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_axis_0_sum_is_bigger_than_one__returns_false():
  qk = np.array([0.5, 0.6], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_axis_1_sum_is_bigger_than_one__returns_false():
  qk = np.array([[0.5, 0.6]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_axis_1_one_is_bigger__returns_false():
  qk = np.array([[1.0], [1.1]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_axis_1_one_sum_is_bigger__returns_false():
  qk = np.array([[0.5, 0.5], [0.5, 0.6]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result
