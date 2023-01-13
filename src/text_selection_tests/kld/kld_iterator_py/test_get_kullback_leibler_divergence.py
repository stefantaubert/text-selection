import numpy as np

from text_selection.kld.kld_iterator import get_kullback_leibler_divergence


# region empty
def test_s0_empty__returns_zero():
  pk = np.array([], dtype=np.float64)
  qk = np.array([], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_s0x0_empty__returns_empty():
  pk = np.empty(shape=(0, 0), dtype=np.float64)
  qk = np.empty(shape=(0, 0), dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(0,), dtype=np.float64))


def test_s1x0_empty__returns_zero():
  pk = np.array([[]], dtype=np.float64)
  qk = np.array([[]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0]))


def test_s2x0_empty__returns_zeros():
  pk = np.array([[], []], dtype=np.float64)
  qk = np.array([[], []], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0, 0.0]))
# endregion


# region equal
def test_s1_equal__returns_zero():
  pk = np.array([1.0], dtype=np.float64)
  qk = np.array([1.0], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_s2_equal__returns_zero():
  pk = np.array([0.5, 0.5], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_1x1_equal__returns_zero():
  pk = np.array([[1.0]], dtype=np.float64)
  qk = np.array([[1.0]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0]))


def test_1x2_equal__returns_zero():
  pk = np.array([[0.5, 0.5]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0]))


def test_2x1_equal__returns_zero():
  pk = np.array([[1.0], [1.0]], dtype=np.float64)
  qk = np.array([[1.0], [1.0]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0, 0.0]))


def test_2x2_equal__returns_zero():
  pk = np.array([[0.5, 0.5], [0.7, 0.3]], dtype=np.float64)
  qk = np.array([[0.5, 0.5], [0.7, 0.3]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0, 0.0]))
# endregion


# region different
def test_s2_different__returns_correct_kld():
  pk = np.array([0.3, 0.7], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.08228287850505178


def test_s1x2_different__returns_correct_kld():
  pk = np.array([[0.3, 0.7]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287850505178]))


def test_s2x2_different__returns_correct_kld():
  pk = np.array([[0.3, 0.7], [0.7, 0.3]], dtype=np.float64)
  qk = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287850505178, 0.08228287850505178]))
# endregion


# region nan
def test_s1_nan__returns_inf():
  pk = np.array([np.nan], dtype=np.float64)
  qk = np.array([1.0], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert np.isinf(result)


def test_s2_nan__returns_inf():
  pk = np.array([np.nan, np.nan], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert np.isinf(result)


def test_s1x1_nan__returns_inf():
  pk = np.array([[np.nan]], dtype=np.float64)
  qk = np.array([[1.0]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([np.inf]))


def test_s1x2_nan__returns_inf():
  pk = np.array([[np.nan, np.nan]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([np.inf]))


def test_s2x1_nan__returns_inf():
  pk = np.array([[np.nan], [np.nan]], dtype=np.float64)
  qk = np.array([[1.0], [1.0]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([np.inf, np.inf]))


def test_s2x2_nan__returns_inf():
  pk = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
  qk = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([np.inf, np.inf]))
# endregion


# region dtypes
def test_s2_float_32__returns_float_32():
  pk = np.array([0.3, 0.7], dtype=np.float32)
  qk = np.array([0.5, 0.5], dtype=np.float32)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  np.testing.assert_almost_equal(result, 0.08228287)
  assert result.dtype == np.float32


def test_s2_float_64__returns_float_64():
  pk = np.array([0.3, 0.7], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.08228287850505178
  assert result.dtype == np.float64


def test_s1x2_float_32__returns_float_32():
  pk = np.array([[0.3, 0.7]], dtype=np.float32)
  qk = np.array([[0.5, 0.5]], dtype=np.float32)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287], dtype=np.float32))
  assert result.dtype == np.float32


def test_s1x2_float_64__returns_float_64():
  pk = np.array([[0.3, 0.7]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287850505178]))
  assert result.dtype == np.float64


def test_s2x2_float_32__returns_float_32():
  pk = np.array([[0.3, 0.7], [0.7, 0.3]], dtype=np.float32)
  qk = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287, 0.08228287], dtype=np.float32))
  assert result.dtype == np.float32


def test_s2x2_float_64__returns_float_64():
  pk = np.array([[0.3, 0.7], [0.7, 0.3]], dtype=np.float64)
  qk = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287850505178, 0.08228287850505178]))
  assert result.dtype == np.float64
# endregion


def test_s3x2_component_test__returns_correct_klds():
  pk = np.array([[0.5, 0.5], [np.nan, np.nan], [0.3, 0.7]], dtype=np.float64)
  qk = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0, np.inf, 0.08228287850505178]))
