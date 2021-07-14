import cProfile
import random
import time
import unittest
from collections import OrderedDict
from typing import List

import numpy as np
from scipy.stats import entropy

from text_selection.greedy_kld_methods import *
from text_selection.utils import get_distribution, get_reverse_distribution

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def get_random_list(length: int, chars: List[str]) -> List[str]:
  res = [random.choice(chars) for _ in range(length)]
  return res


class UnitTests(unittest.TestCase):

  def test_get_distribution(self):
    data = {
      2: [1, 2, 3],
      3: [1, 2],
      4: [1],
    }

    x = get_distribution(data)

    assert_res = {
      1: 3 / 6,
      2: 2 / 6,
      3: 1 / 6,
    }

    self.assertEqual(assert_res, x)

  def test_get_reverse_distribution(self):
    data = {
      2: [1, 2, 3],
      3: [1, 2],
      4: [1],
    }

    x = get_reverse_distribution(data)

    assert_res = {
      1: 1 / 6,
      2: 2 / 6,
      3: 3 / 6,
    }

    self.assertEqual(assert_res, x)

  def test_entropy(self):
    dist_1 = OrderedDict({
      ("Hallo", "h"): 1 / 7,
      ("du", "d"): 3 / 7,
      ("und", "u"): 2 / 7,
      ("Bye", "b"): 1 / 7
    })

    dist_2 = OrderedDict({
      ("Hallo", "h"): 0.2,
      ("du", "d"): 0.3,
      ("und", "u"): 0.4,
      ("Bye", "b"): 0.1
    })

    res = entropy(list(dist_1.values()), list(dist_2.values()))

    right_div = 1 / 7 * np.log((1 / 7) / 0.2) + 3 / 7 * np.log((3 / 7) / 0.3) + \
        2 / 7 * np.log((2 / 7) / 0.4) + 1 / 7 * np.log((1 / 7) / 0.1)

    self.assertAlmostEqual(right_div, res)

  def test_kullback_leiber__same_dist__expect_zero(self):
    dist_1 = OrderedDict({
      ("Hallo", "h"): 1 / 7,
      ("du", "d"): 3 / 7,
      ("und", "u"): 2 / 7,
      ("Bye", "b"): 1 / 7
    })

    res = entropy(list(dist_1.values()), list(dist_1.values()))

    self.assertEqual(0, res)

  def test_greedy__works(self):
    target_dist = {
      ("Hallo", "h"): 0.2,
      ("du", "d"): 0.3,
      ("und", "u"): 0.4,
      ("Bye", "b"): 0.1,
      ("Irrelevante", "i"): 0,
      ("Worte", "w"): 0
    }

    data = OrderedDict({
      1: [("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
      2: [("Bye", "b"), ("und", "u"), ("du", "d")],
      3: [("Irrelevante", "i"), ("Worte", "w")],
    })

    res = sort_greedy_kld(data, target_dist)

    self.assertEqual(OrderedSet([1, 2, 3]), res)

  def test_sort_greedy_kld(self):
    data = OrderedDict({
      2: [2, 3],
      3: [1, 2, 3],
      4: [1, 2, 3, 4],
      5: [1, 2, 3, 4, 4],
    })

    distr = {
      1: 0.25,
      2: 0.25,
      3: 0.25,
      4: 0.25,
    }

    res = sort_greedy_kld(data, distr)

    self.assertEqual(OrderedSet([4, 5, 3, 2]), res)

  def test_performance(self):
    n_data = 500
    data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})

    distr = get_uniform_distribution(data)

    start = time.perf_counter()

    with cProfile.Profile() as pr:
        # ... do something ...
      res = sort_greedy_kld(data, distr)
    pr.print_stats()
    end = time.perf_counter()
    duration = end - start

    self.assertTrue(duration < 6)

  def test_performance_its(self):
    n_data = 500
    data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})

    distr = get_uniform_distribution(data)

    start = time.perf_counter()

    with cProfile.Profile() as pr:
        # ... do something ...
      res = sort_greedy_kld_iterations(data, distr, n_data - 1)
    pr.print_stats()
    end = time.perf_counter()
    duration = end - start

    self.assertTrue(duration < 6)

  def test_performance_until(self):
    n_data = 500
    data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})
    until_values = {i: 1 for i in range(n_data)}

    distr = get_uniform_distribution(data)

    start = time.perf_counter()

    with cProfile.Profile() as pr:
        # ... do something ...
      res = sort_greedy_kld_until(data, distr, until_values, 499)
    pr.print_stats()
    end = time.perf_counter()
    duration = end - start

    self.assertTrue(duration < 6)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
