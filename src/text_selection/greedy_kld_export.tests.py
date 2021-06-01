import unittest

from ordered_set import OrderedSet
from text_selection.greedy_kld_export import \
    greedy_kld_uniform_ngrams_default
from text_selection.utils import *


class UnitTests(unittest.TestCase):
  def test_greedy_kld_uniform_ngrams__one_gramas__selects_best_fitting_first(self):
    data = OrderedDict({
      1: ["a", "a"],  # one new
      2: ["c", "a"],  # two new
      3: ["a", "a"],  # one new
      4: ["c", "a"],  # two new
    })

    res = greedy_kld_uniform_ngrams_default(
      data=data,
      n_gram=1,
      ignore_symbols=None,
    )

    self.assertEqual(OrderedSet([2, 4, 1, 3]), res)

  def test_greedy_kld_uniform_ngrams__two_grams_all_equal__return_same_order(self):
    data = OrderedDict({
      1: ["a", "a"],  # one new
      2: ["a", "b"],  # one new
      3: ["c", "a"],  # one new
    })

    res = greedy_kld_uniform_ngrams_default(
      data=data,
      n_gram=2,
      ignore_symbols=None,
    )

    self.assertEqual(OrderedSet([1, 2, 3]), res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
