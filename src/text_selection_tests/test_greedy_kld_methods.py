import cProfile
import math
import random
import time
from collections import OrderedDict
from logging import getLogger
from typing import List

import numpy as np
from ordered_set import OrderedSet
from scipy.stats import entropy

from text_selection.greedy_kld_methods import (__get_distribution, dict_to_array_ordered,
                                               get_available_arrays, get_divergences,
                                               get_keys_sort_after_value,
                                               get_smallest_divergence_keys,
                                               get_uniform_distribution, merge_arrays,
                                               sort_greedy_kld, sort_greedy_kld_iterations,
                                               sort_greedy_kld_until,
                                               sort_greedy_kld_until_with_preselection,
                                               sort_kld_parts, split_into_equal_parts,
                                               sync_dict_keys_to_keys_inplace)

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def get_random_list(length: int, chars: List[str]) -> List[str]:
  res = [random.choice(chars) for _ in range(length)]
  return res


def test_get_divergences_mp():
  data = OrderedDict({
    0: np.array([1, 0]),
    1: np.array([0, 1]),
  })

  covered_counts = np.array([1, 0])
  target_distr = np.array([0.5, 0.5])

  result = get_divergences(
    data, covered_counts, target_distr,
    n_jobs=16,
    maxtasksperchild=None,
    chunksize=1,
  )

  assert result.keys() == {0, 1}
  assert result[0] == 0.6931471805599453
  assert result[1] == 0.0


def xtest_get_divergences_mp__stress_test():
  np.random.seed(1111)
  utterance_count = 1000000
  utterance_len = 100
  max_ngram = 100
  data = OrderedDict({
    k + 1: np.array(np.random.randint(0, max_ngram, utterance_len)) for k in range(utterance_count)
  })

  covered_counts = np.array([0] * max_ngram)
  target_distr = np.array([1 / max_ngram] * max_ngram)

  start = time.perf_counter()
  result = get_divergences(
    data, covered_counts, target_distr,
    n_jobs=16,
    maxtasksperchild=None,
    chunksize=math.ceil(utterance_count / 16),
  )

  end = time.perf_counter()
  duration = end - start
  logger = getLogger(__name__)
  logger.info(f"Duration: {duration}")

  assert 0 not in result
  assert result[1] == 0.23035848513245613
  assert len(result) == utterance_count
  assert 4 < duration < 6


def test_get_distribution__empty_input():
  counts = np.array([])
  result = __get_distribution(counts)

  assert len(result) == 0


def test_get_distribution__returns_distribution():
  counts = np.array([3, 2, 1])
  result = __get_distribution(counts)

  assert len(result) == 3
  assert result[0] == 3 / 6
  assert result[1] == 2 / 6
  assert result[2] == 1 / 6


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

  res = sort_greedy_kld(data, target_dist, n_jobs=1, maxtasksperchild=None, chunksize=1)

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

  res = sort_greedy_kld(data, distr, n_jobs=1, maxtasksperchild=None, chunksize=1)

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
    n_jobs=1, maxtasksperchild=None, chunksize=1,
  )

  assert OrderedSet([4, 5]) == res


def test_sync_dict_keys_to_keys__keeps_unchanged():
  counter = {
    1: 5,
    2: 5,
  }
  keys = {1, 2}

  sync_dict_keys_to_keys_inplace(counter, keys)

  assert counter.keys() == {1, 2}
  assert counter[1] == 5
  assert counter[2] == 5


def test_sync_dict_keys_to_keys__adds_key():
  counter = {
    1: 5,
  }
  keys = {1, 2}

  sync_dict_keys_to_keys_inplace(counter, keys)

  assert counter.keys() == {1, 2}
  assert counter[1] == 5
  assert counter[2] == 0


def test_sync_dict_keys_to_keys__removes_key():
  counter = {
    1: 5,
    2: 5,
  }
  keys = {1}

  sync_dict_keys_to_keys_inplace(counter, keys)

  assert counter.keys() == {1}
  assert counter[1] == 5


def test_get_available_arrays__empty_input():
  data = dict()

  result = get_available_arrays(
    data=data,
    target_symbols_ordered=OrderedSet([1, 2, 3])
  )

  assert len(result) == 0


def test_get_available_arrays():
  data = {
    2: [1, 2, 3],
    3: [2, 3, 4],
    4: [3, 4, 5],
  }

  result = get_available_arrays(
    data=data,
    target_symbols_ordered=OrderedSet([1, 2, 3])
  )

  assert len(result) == 3
  assert list(result.keys()) == [2, 3, 4]
  assert list(result[2]) == [1, 1, 1]
  assert list(result[3]) == [0, 1, 1]
  assert list(result[4]) == [0, 0, 1]


def xtest_performance():
  n_data = 500
  data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})

  distr = get_uniform_distribution(data)

  with cProfile.Profile() as pr:
      # ... do something ...
    start = time.perf_counter()
    res = sort_greedy_kld(data, distr, n_jobs=1, maxtasksperchild=None, chunksize=500)
    end = time.perf_counter()
  pr.print_stats()
  duration = end - start

  #assert duration < 6
  assert duration < 14


def test_get_uniform_distribution__empty_input():
  data = OrderedDict()

  result = get_uniform_distribution(data)

  assert len(result) == 0


def test_get_uniform_distribution__detects_all_keys():
  data = OrderedDict({
    0: [0, 3],
    1: [5],
  })

  result = get_uniform_distribution(data)

  assert len(result) == 3
  assert result[0] == 1 / 3
  assert result[3] == 1 / 3
  assert result[5] == 1 / 3


def test_dict_to_array_ordered_after_keys__empty_input():
  data = {}
  order = OrderedSet()

  result = dict_to_array_ordered(data, order)

  assert len(result) == 0


def test_dict_to_array_ordered_after_keys__is_sorted():
  data = {
    3: [4],
    1: [5],
    2: [6],
  }
  order = OrderedSet([1, 2, 3])

  result = dict_to_array_ordered(data, order)

  assert len(result) == 3
  assert result[0] == [5]
  assert result[1] == [6]
  assert result[2] == [4]


def xtest_performance_its():
  n_data = 500
  data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})

  distr = get_uniform_distribution(data)

  start = time.perf_counter()

  with cProfile.Profile() as pr:
    res = sort_greedy_kld_iterations(
      data, distr, n_data - 1, n_jobs=1, maxtasksperchild=None, chunksize=1)
  pr.print_stats()
  end = time.perf_counter()
  duration = end - start

  assert duration < 6


def xtest_performance_until():
  n_data = 500
  data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})
  until_values = {i: 1 for i in range(n_data)}

  distr = get_uniform_distribution(data)

  start = time.perf_counter()

  with cProfile.Profile() as pr:
    res = sort_greedy_kld_until(data, distr, until_values, 499,
                                n_jobs=1, maxtasksperchild=None, chunksize=1)
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
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
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
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
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
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([2]) == res


def test_sort_greedy_kld_until_with_preselection__multiple_candidates_key_order_is_preserved():
  preselection = OrderedDict()

  data = OrderedDict({
    7: ("i"),
    8: ("k",),
    3: ("k",),
  })

  distr = {
    "k": 1.0,
  }

  until_values = {
    7: 1,
    8: 1,
    3: 1,
  }

  res = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=distr,
    until_values=until_values,
    until_value=1,
    preselection=preselection,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([8]) == res


def test_sort_greedy_kld_until_with_preselection__too_few_data():
  preselection = OrderedDict()
  data = OrderedDict({
    5: ["a"],
  })

  durations = {
    5: 1.0,
  }

  target_duration = 2.0

  target_distribution = {
    "a": 1.0,
  }

  res = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=target_distribution,
    until_values=durations,
    until_value=target_duration,
    preselection=preselection,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([5]) == res


def test_sort_greedy_kld_until_with_preselection__empty_input():
  preselection = OrderedDict()
  data = OrderedDict()

  distr = {}

  until_values = {
    "2": 1,
  }

  res = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=distr,
    until_values=until_values,
    until_value=2,
    preselection=preselection,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet() == res


def test_sort_greedy_kld_until_with_preselection__irrelevant_ngrams_were_ignored():
  preselection = OrderedDict()
  data = OrderedDict({
    5: ["a", "b"],
    6: ["b"],
    7: ["c"],
  })

  durations = {
    5: 1.0,
    6: 1.0,
    7: 1.0,
  }

  target_duration = 2.0

  target_distribution = {
    "a": 1.0,
    "c": 1.0,
  }

  res = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=target_distribution,
    until_values=durations,
    until_value=target_duration,
    preselection=preselection,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([5, 7]) == res


def test_sort_greedy_kld_until_with_preselection__warning_on_not_existing_ngrams_compared_to_targed_distr():
  preselection = OrderedDict({
    4: ["a"],
  })

  data = OrderedDict({
    5: ["b"],
  })

  durations = {
    5: 1.0,
  }

  target_duration = 1.0

  target_distribution = {
    "a": 1.0,
    "b": 1.0,
    "c": 1.0,
  }

  res = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=target_distribution,
    until_values=durations,
    until_value=target_duration,
    preselection=preselection,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([5]) == res


def test_get_smallest_divergence__one_entry_returns_this_entry():
  divergences = OrderedDict({1: 0.5})

  result = get_smallest_divergence_keys(divergences)

  assert result == OrderedSet([1])


def test_get_smallest_divergence__two_same_entries_returns_both_entries():
  divergences = OrderedDict({2: 0.5, 1: 0.5})
  result = get_smallest_divergence_keys(divergences)

  assert result == OrderedSet([2, 1])


def test_get_smallest_divergence__two_different_entries_returns_the_smallest_one():
  divergences = OrderedDict({2: 0.5, 1: 0.4})
  result = get_smallest_divergence_keys(divergences)

  assert result == OrderedSet([1])


def test_sort_kld_parts__one_part():
  data = {
    5: ["a"],
  }

  lengths = OrderedDict({
    5: 1,
  })

  target_distribution = {
    "a": 1.0,
  }

  res = sort_kld_parts(
    data=data,
    target_dist=target_distribution,
    take_per_part=1,
    lengths=lengths,
    parts_count=1,
    n_jobs=1, maxtasksperchild=None, chunksize=1,
  )

  assert OrderedSet([5]) == res


def test_sort_kld_parts__two_parts_take_one():
  data = {
    1: ["a"],  # part 1
    2: ["a"],  # part 1
    3: ["b"],  # part 2
    4: ["b"],  # part 2
  }

  lengths = OrderedDict({
    1: 1,
    2: 2,
    3: 3,
    4: 4,
  })

  target_distribution = {
    "a": 0.5,
    "b": 0.5,
  }

  res = sort_kld_parts(
    data=data,
    target_dist=target_distribution,
    take_per_part=1,
    lengths=lengths,
    parts_count=2,
    n_jobs=1, maxtasksperchild=None, chunksize=1,
  )

  assert OrderedSet([1, 3, 2, 4]) == res


def test_sort_kld_parts__two_parts_take_two():
  data = {
    1: ["a"],  # part 1
    2: ["a"],  # part 1
    3: ["b"],  # part 2
    4: ["b"],  # part 2
  }

  lengths = OrderedDict({
    1: 1,
    2: 2,
    3: 3,
    4: 4,
  })

  target_distribution = {
    "a": 0.5,
    "b": 0.5,
  }

  res = sort_kld_parts(
    data=data,
    target_dist=target_distribution,
    take_per_part=2,
    lengths=lengths,
    parts_count=2,
    n_jobs=1, maxtasksperchild=None, chunksize=1,
  )

  assert OrderedSet([1, 2, 3, 4]) == res


def test_sort_kld_parts__two_parts_take_three__skips_non_existing_third():
  data = {
    1: ["a"],  # part 1
    2: ["b"],  # part 1
    3: ["b"],  # part 2
    4: ["a"],  # part 2
  }

  lengths = OrderedDict({
    1: 1,
    2: 2,
    3: 3,
    4: 4,
  })

  target_distribution = {
    "a": 0.5,
    "b": 0.5,
  }

  res = sort_kld_parts(
    data=data,
    target_dist=target_distribution,
    take_per_part=3,
    lengths=lengths,
    parts_count=2,
    n_jobs=1, maxtasksperchild=None, chunksize=1,
  )

  assert OrderedSet([1, 2, 3, 4]) == res


def test_sort_kld_parts__five_parts_take_one__ignores_fifth_part():
  data = {
    1: ["a"],  # part 1
    2: ["a"],  # part 2
    3: ["b"],  # part 3
    4: ["b"],  # part 4
  }

  lengths = OrderedDict({
    1: 1,
    2: 2,
    3: 3,
    4: 4,
  })

  target_distribution = {
    "a": 0.5,
    "b": 0.5,
  }

  res = sort_kld_parts(
    data=data,
    target_dist=target_distribution,
    take_per_part=1,
    lengths=lengths,
    parts_count=5,
    n_jobs=1, maxtasksperchild=None, chunksize=1,
  )

  assert OrderedSet([1, 2, 3, 4]) == res


def test_sort_kld_parts__one_part_take_one__makes_four_rounds():
  data = {
    1: ["a"],  # part 1
    2: ["a"],  # part 1
    3: ["b"],  # part 1
    4: ["b"],  # part 1
  }

  lengths = OrderedDict({
    1: 1,
    2: 2,
    3: 3,
    4: 4,
  })

  target_distribution = {
    "a": 0.5,
    "b": 0.5,
  }

  res = sort_kld_parts(
    data=data,
    target_dist=target_distribution,
    take_per_part=1,
    lengths=lengths,
    parts_count=1,
    n_jobs=1, maxtasksperchild=None, chunksize=1,
  )

  assert OrderedSet([1, 3, 2, 4]) == res


def test_sort_after_value__empty_input():
  lengths = OrderedDict()

  result = get_keys_sort_after_value(lengths)

  assert result == OrderedSet()


def test_sort_after_value__one_entry():
  lengths = OrderedDict({
    2: 5,
  })

  result = get_keys_sort_after_value(lengths)

  assert result == OrderedSet([2])


def test_sort_after_value_two_distinct_entries():
  lengths = OrderedDict({
    2: 5,
    1: 4,
  })

  result = get_keys_sort_after_value(lengths)

  assert result == OrderedSet([1, 2])


def test_sort_after_value_two_equal_entries__returns_first_in_dict():
  lengths = OrderedDict({
    2: 5,
    1: 5,
  })

  result = get_keys_sort_after_value(lengths)

  assert result == OrderedSet([2, 1])


def test_split_into_equal_parts__empty_set():
  keys = OrderedSet()

  result = split_into_equal_parts(keys, parts_count=1)

  assert result == []


def test_split_into_equal_parts__one_entry_one_part():
  keys = OrderedSet([1])

  result = split_into_equal_parts(keys, parts_count=1)

  assert len(result) == 1
  assert result[0] == OrderedSet([1])


def test_split_into_equal_parts__one_entry_two_parts__returns_one_part():
  keys = OrderedSet([1])

  result = split_into_equal_parts(keys, parts_count=2)

  assert len(result) == 1
  assert result[0] == OrderedSet([1])


def test_split_into_equal_parts__two_entries_two_parts():
  keys = OrderedSet([1, 2])

  result = split_into_equal_parts(keys, parts_count=2)

  assert len(result) == 2
  assert result[0] == OrderedSet([1])
  assert result[1] == OrderedSet([2])


def test_split_into_equal_parts__three_entries_two_parts__first_part_gets_two_entries():
  keys = OrderedSet([1, 2, 3])

  result = split_into_equal_parts(keys, parts_count=2)

  assert len(result) == 2
  assert result[0] == OrderedSet([1, 2])
  assert result[1] == OrderedSet([3])


def test_split_into_equal_parts__five_entries_three_parts__first_two_parts_get_two_entries():
  keys = OrderedSet([1, 2, 3, 4, 5])

  result = split_into_equal_parts(keys, parts_count=3)

  assert len(result) == 3
  assert result[0] == OrderedSet([1, 2])
  assert result[1] == OrderedSet([3, 4])
  assert result[2] == OrderedSet([5])
