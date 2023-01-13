
import numpy as np
from ordered_set import OrderedSet

from text_selection.common.ngram_extractor import get_count_array


def test_component():
  result = get_count_array(
    ngram_nrs=(1, 2, 3, 1, 3, 3, 4),
    target_symbols_ordered=OrderedSet((1, 2, 3)),
  )
  np.testing.assert_array_equal(result, np.array([2, 1, 3]))


def test_empty_3_targets__returns_3_zero():
  result = get_count_array(
    ngram_nrs=tuple(),
    target_symbols_ordered=OrderedSet((1, 2, 3)),
  )
  np.testing.assert_array_equal(result, np.array([0, 0, 0]))


def test_empty_0_targets__returns_empty():
  result = get_count_array(
    ngram_nrs=tuple(),
    target_symbols_ordered=OrderedSet(),
  )
  np.testing.assert_array_equal(result, np.array([]))


def test_2_different_numbers_1_target__returns_1_target_count():
  result = get_count_array(
    ngram_nrs=(1, 2),
    target_symbols_ordered=OrderedSet((2,)),
  )
  np.testing.assert_array_equal(result, np.array([1]))
