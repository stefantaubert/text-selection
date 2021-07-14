import unittest

from text_selection.random_export import *
from text_selection.utils import *


class UnitTests(unittest.TestCase):
  def test_random_default__returns_random(self):
    data = OrderedDict({
      1: ["a", "a"],  # one new
      2: ["c", "a"],  # two new
      3: ["d", "a"],  # two new
    })

    res = random_default(
      data=data,
      seed=1,
    )

    assert_res = OrderedSet([2, 3, 1])
    self.assertEqual(assert_res, res)

  def test_random_ngrams_default_cover__returns_covered_random(self):
    data = OrderedDict({
      1: ["a"],
      2: ["a"],
      3: ["c"],
      4: ["a"],
      5: ["d"],
    })

    seed_tries = range(500)

    for seed in seed_tries:
      res = random_ngrams_cover_default(
        data=data,
        seed=seed,
        ignore_symbols=None,
        n_gram=1,
      )

      entries = [data[x][0] for x in res]

      self.assertEqual({"a", "c", "d"}, set(entries[:3]))
      self.assertEqual(5, len(res))

  def test_random_ngrams_cover_seconds__one_gram_two_seconds__return_correct_sorting(self):
    data = OrderedDict({
      1: ["a", "a"],  # one new
      2: ["c", "a"],  # two new
      3: ["d", "a"],  # two new
    })

    durations = {
      1: 1,
      2: 1,
      3: 1,
    }

    res = random_ngrams_cover_seconds(
      data=data,
      n_gram=1,
      ignore_symbols=None,
      durations_s=durations,
      seconds=2,
      seed=1,
    )

    assert_res = OrderedSet([1, 2])
    self.assertEqual(assert_res, res)

  def test_get_n_divergent_random_seconds__one_entry__return_correct_sorting(self):
    data = OrderedDict({
      1: [],
      2: [],
      3: [],
    })

    durations = {
      1: 1,
      2: 1,
      3: 1,
      4: 1,
      5: 1,
    }

    res = n_divergent_random_seconds(
      data=data,
      durations_s=durations,
      seconds=1,
      seed=1,
      n=3,
    )

    assert_res = [
      OrderedSet([2]),
      OrderedSet([3]),
      OrderedSet([1]),
    ]

    self.assertEqual(assert_res, res)

  def test_get_n_divergent_random_seconds__two_entries__return_correct_sorting(self):
    data = OrderedDict({
      1: [],
      2: [],
      3: [],
    })

    durations = {
      1: 1,
      2: 1,
      3: 1,
      4: 1,
      5: 1,
    }

    res = n_divergent_random_seconds(
      data=data,
      durations_s=durations,
      seconds=2,
      seed=1,
      n=3,
    )

    assert_res = [
      OrderedSet([2, 3]),
      OrderedSet([3, 1]),
      OrderedSet([1, 2]),
    ]

    self.assertEqual(assert_res, res)

  def test_get_n_divergent_random_seconds__three_entries__return_correct_sorting(self):
    data = OrderedDict({
      1: [],
      2: [],
      3: [],
    })

    durations = {
      1: 1,
      2: 1,
      3: 1,
      4: 1,
      5: 1,
    }

    res = n_divergent_random_seconds(
      data=data,
      durations_s=durations,
      seconds=3,
      seed=1,
      n=3,
    )

    assert_res = [
      OrderedSet([2, 3, 1]),
      OrderedSet([3, 1, 2]),
      OrderedSet([1, 2, 3]),
    ]

    self.assertEqual(assert_res, res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
