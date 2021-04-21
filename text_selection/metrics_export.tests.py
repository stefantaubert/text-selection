import math
import unittest
from collections import OrderedDict

from text_selection.metrics_export import get_rarity_ngrams


class UnitTests(unittest.TestCase):
  def test_get_rarity_ngrams__one_entry(self):
    data = OrderedDict({
      1: ["a", "b"],  # one new
    })

    corpus = OrderedDict({
      1: ["a", "x"],
      2: ["b", "b"],
      3: ["c", "c"],
      4: ["d", "d"],
    })

    res = get_rarity_ngrams(
      data=data,
      corpus=corpus,
      n_gram=1,
      ignore_symbols=None,
    )

    assert_res = OrderedDict({
      1: (1 / 8 + 2 / 8) / 2,  # a + b
    })

    self.assertEqual(assert_res, res)

  def test_get_rarity_ngrams__two_entries(self):
    data = OrderedDict({
      1: ["a", "b"],  # one new
      3: ["c", "c"],  # one new
    })

    corpus = OrderedDict({
      1: ["a", "x"],
      2: ["b", "b"],
      3: ["c", "c"],
      4: ["d", "d"],
    })

    res = get_rarity_ngrams(
      data=data,
      corpus=corpus,
      n_gram=1,
      ignore_symbols=None,
    )

    assert_res = OrderedDict({
      1: (1 / 8 + 2 / 8) / 2,  # a + b
      3: (2 / 8 + 2 / 8) / 2,  # c + c
    })

    self.assertEqual(assert_res, res)

  def test_get_rarity_ngrams__not_existing__has_zero(self):
    data = OrderedDict({
      1: ["z", "a"],  # one new
    })

    corpus = OrderedDict({
      1: ["a", "x"],
      2: ["b", "b"],
      3: ["c", "c"],
      4: ["d", "d"],
    })

    res = get_rarity_ngrams(
      data=data,
      corpus=corpus,
      n_gram=1,
      ignore_symbols=None,
    )

    assert_res = OrderedDict({
      1: (0 + 1 / 8) / 2,  # z + a
    })

    self.assertEqual(assert_res, res)

  def test_get_rarity_ngrams__one_ignored_in_data__ignores_it(self):
    data = OrderedDict({
      1: ["z", "a"],  # one new
    })

    corpus = OrderedDict({
      1: ["a", "x"],
      2: ["b", "b"],
      3: ["c", "c"],
      4: ["d", "d"],
    })

    res = get_rarity_ngrams(
      data=data,
      corpus=corpus,
      n_gram=1,
      ignore_symbols={"z"},
    )

    assert_res = OrderedDict({
      1: 1 / 8,  # a
    })

    self.assertEqual(assert_res, res)

  def test_get_rarity_ngrams__one_ignored_in_corpus__ignores_it(self):
    data = OrderedDict({
      1: ["x", "a"],  # one new
    })

    corpus = OrderedDict({
      1: ["a", "x"],
      2: ["b", "b"],
      3: ["c", "c"],
      4: ["d", "d"],
    })

    res = get_rarity_ngrams(
      data=data,
      corpus=corpus,
      n_gram=1,
      ignore_symbols={"x"},
    )

    assert_res = OrderedDict({
      1: 1 / 7,  # a
    })

    self.assertEqual(assert_res, res)

  def test_get_rarity_ngrams__empty_entry__returns_inf(self):
    data = OrderedDict({
      1: [],
    })

    corpus = OrderedDict({
      1: ["a", "x"],
      2: ["b", "b"],
      3: ["c", "c"],
      4: ["d", "d"],
    })

    res = get_rarity_ngrams(
      data=data,
      corpus=corpus,
      n_gram=1,
      ignore_symbols=None,
    )

    assert_res = OrderedDict({
      1: math.inf,
    })

    self.assertEqual(assert_res, res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
