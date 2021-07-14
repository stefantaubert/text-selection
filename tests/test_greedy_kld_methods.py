import cProfile
import random
import time
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


def test_get_distribution():
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

  assert assert_res == x


def test_get_reverse_distribution():
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

  assert assert_res == x


def test_entropy():
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

  assert round(abs(right_div - res), 7) == 0


def test_kullback_leiber__same_dist__expect_zero():
  dist_1 = OrderedDict({
    ("Hallo", "h"): 1 / 7,
    ("du", "d"): 3 / 7,
    ("und", "u"): 2 / 7,
    ("Bye", "b"): 1 / 7
  })

  res = entropy(list(dist_1.values()), list(dist_1.values()))

  assert 0 == res


def test_greedy__works():
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

  assert OrderedSet([1, 2, 3]) == res


def test_sort_greedy_kld():
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

  assert OrderedSet([4, 5, 3, 2]) == res


def test_merge_arrays():
  arrays = {
    0: np.array([0, 1, 2, 3, 4, 5]),
    1: np.array([0, 1, 2, 3, 4, 5]),
    2: np.array([0, 1, 2, 3, 4, 5]),
  }
  result = merge_arrays(arrays)

  assert list(result) == [0, 3, 6, 9, 12, 15]


def test_sort_greedy_kld_until():
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

  until_values = {
    2: 1,
    3: 0.9,
    4: 0.2,
    5: 1,
  }

  res = sort_greedy_kld_until(
    data=data,
    target_dist=distr,
    until_values=until_values,
    until_value=2,
  )

  assert OrderedSet([4, 5]) == res


def test_sync_dict_keys_to_keys__keeps_unchanged():
  counter = {
    1: 5,
    2: 5,
  }
  keys = {1, 2}

  sync_dict_keys_to_keys(counter, keys)

  assert counter.keys() == {1, 2}
  assert counter[1] == 5
  assert counter[2] == 5


def test_sync_dict_keys_to_keys__adds_key():
  counter = {
    1: 5,
  }
  keys = {1, 2}

  sync_dict_keys_to_keys(counter, keys)

  assert counter.keys() == {1, 2}
  assert counter[1] == 5
  assert counter[2] == 0


def test_sync_dict_keys_to_keys__removes_key():
  counter = {
    1: 5,
    2: 5,
  }
  keys = {1}

  sync_dict_keys_to_keys(counter, keys)

  assert counter.keys() == {1}
  assert counter[1] == 5


def test_get_available_arrays__empty_input():
  data = OrderedDict()

  result = get_available_arrays(
    data=data,
    all_keys={1, 2, 3},
  )

  assert isinstance(result, OrderedDict)
  assert len(result) == 0


def test_get_available_arrays():
  data = OrderedDict({
    2: [1, 2, 3],
    3: [2, 3, 4],
    4: [3, 4, 5],
  })

  result = get_available_arrays(
    data=data,
    all_keys={1, 2, 3},
  )

  assert isinstance(result, OrderedDict)
  assert len(result) == 3
  assert list(result.keys()) == [2, 3, 4]
  assert list(result[2]) == [1, 1, 1]
  assert list(result[3]) == [0, 1, 1]
  assert list(result[4]) == [0, 0, 1]


def test_performance():
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

  assert duration < 6


def test_performance_its():
  n_data = 500
  data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})

  distr = get_uniform_distribution(data)

  start = time.perf_counter()

  with cProfile.Profile() as pr:
    res = sort_greedy_kld_iterations(data, distr, n_data - 1)
  pr.print_stats()
  end = time.perf_counter()
  duration = end - start

  assert duration < 6


def test_performance_until():
  n_data = 500
  data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})
  until_values = {i: 1 for i in range(n_data)}

  distr = get_uniform_distribution(data)

  start = time.perf_counter()

  with cProfile.Profile() as pr:
    res = sort_greedy_kld_until(data, distr, until_values, 499)
  pr.print_stats()
  end = time.perf_counter()
  duration = end - start

  assert duration < 6


def test_sort_greedy_kld_until_with_preselection__one_preselected():
  preselection = OrderedDict({
    1: [1],
  })

  data = OrderedDict({
    2: [1],
    3: [2],
  })

  distr = {
    1: 0.5,
    2: 0.5,
  }

  until_values = {
    2: 1,
    3: 1,
  }

  res = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=distr,
    until_values=until_values,
    until_value=1,
    preselection=preselection,
  )

  assert OrderedSet([3]) == res


def test_sort_greedy_kld_until_with_preselection__one_preselected_but_none_of_the_target_symbols():
  preselection = OrderedDict({
    1: [1],
  })

  data = OrderedDict({
    2: [1],
    3: [2],
  })

  distr = {
    1: 0.5,
    2: 0.5,
  }

  until_values = {
    2: 1,
    3: 1,
  }

  res = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=distr,
    until_values=until_values,
    until_value=1,
    preselection=preselection,
  )

  assert OrderedSet([3]) == res


def test_sort_greedy_kld_until_with_preselection__nothing_preselected():
  preselection = OrderedDict()

  data = OrderedDict({
    2: [1],
    3: [2],
  })

  distr = {
    1: 0.5,
    2: 0.5,
  }

  until_values = {
    2: 1,
    3: 1,
  }

  res = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=distr,
    until_values=until_values,
    until_value=1,
    preselection=preselection,
  )

  assert OrderedSet([2]) == res
