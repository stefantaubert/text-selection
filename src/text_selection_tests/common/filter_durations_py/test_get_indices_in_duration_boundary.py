import numpy as np

from text_selection.common.filter_durations import get_indices_in_duration_boundary


def test_empty():
  result = get_indices_in_duration_boundary(
      data=np.empty(shape=(0,)),
      min_duration_incl=0.5,
      max_duration_excl=2,
    )

  np.testing.assert_array_equal(result, np.empty(shape=(0,)))


def test_min_is_inclusive():
  result = get_indices_in_duration_boundary(
      data=np.array([0, 1, 2]),
      min_duration_incl=1,
      max_duration_excl=np.inf,
    )

  np.testing.assert_array_equal(result, np.array([1, 2]))


def test_max_is_exclusive():
  result = get_indices_in_duration_boundary(
      data=np.array([0, 1, 2]),
      min_duration_incl=0,
      max_duration_excl=1,
    )

  np.testing.assert_array_equal(result, np.array([0]))


def test_componenttest():
  result = get_indices_in_duration_boundary(
      data=np.array([0, 0.5, 1, 2, 3]),
      min_duration_incl=0.5,
      max_duration_excl=2,
    )

  np.testing.assert_array_equal(result, np.array([1, 2]))
