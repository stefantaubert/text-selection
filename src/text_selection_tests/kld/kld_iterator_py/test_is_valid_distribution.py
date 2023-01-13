import numpy as np

from text_selection.kld.kld_iterator import is_valid_distribution


def test_s0_empty_array__returns_true():
  qk = np.empty(shape=(0,), dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert result


def test_s0x0_empty__returns_true():
  qk = np.empty(shape=(0, 0), dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_s1x0_empty_array__returns_true():
  qk = np.empty(shape=(1, 0), dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_s1_non_empty__returns_true():
  qk = np.array([1.0], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert result


def test_s1x1_non_empty__returns_true():
  qk = np.array([[1.0]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_s2_sum__returns_true():
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert result


def test_s1x2_sum__returns_true():
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_s2x2_multi_sum__returns_true():
  qk = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert result


def test_s1_bigger_than_one__returns_false():
  qk = np.array([1.1], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_s1x1_bigger_than_one__returns_false():
  qk = np.array([[1.1]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_s1_smaller_than_one__returns_false():
  qk = np.array([0.9], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_s1_zero__returns_false():
  qk = np.array([0.0], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_s1x1_zero__returns_false():
  qk = np.array([[0.0]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_s1_negative__returns_false():
  qk = np.array([-0.1], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_s1x1_negative__returns_false():
  qk = np.array([[-0.1]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_s1_nan__returns_false():
  qk = np.array([np.nan], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_s1x1_nan__returns_false():
  qk = np.array([[np.nan]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_s1_inf__returns_false():
  qk = np.array([np.inf], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_s1x1_inf__returns_false():
  qk = np.array([[np.inf]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_s2_sum_is_bigger_than_one__returns_false():
  qk = np.array([0.5, 0.6], dtype=np.float64)
  result = is_valid_distribution(qk, axis=0)
  assert not result


def test_s1x2_sum_is_bigger_than_one__returns_false():
  qk = np.array([[0.5, 0.6]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_s2x1_one_is_bigger__returns_false():
  qk = np.array([[1.0], [1.1]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result


def test_s2x2_one_sum_is_bigger__returns_false():
  qk = np.array([[0.5, 0.5], [0.5, 0.6]], dtype=np.float64)
  result = is_valid_distribution(qk, axis=1)
  assert not result
