from ordered_set import OrderedSet

from text_selection.greedy_kld_export import (greedy_kld_uniform_ngrams_default,
                                              greedy_kld_uniform_ngrams_parts,
                                              greedy_kld_uniform_ngrams_seconds_with_preselection)
from text_selection.utils import *


def test_greedy_kld_uniform_ngrams__one_gramas__selects_best_fitting_first():
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
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([2, 4, 1, 3]) == res


def test_greedy_kld_uniform_ngrams__two_grams_all_equal__return_same_order():
  data = OrderedDict({
    1: ["a", "a"],  # one new
    2: ["a", "b"],  # one new
    3: ["c", "a"],  # one new
  })

  res = greedy_kld_uniform_ngrams_default(
    data=data,
    n_gram=2,
    ignore_symbols=None,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([1, 2, 3]) == res


def test_sort_greedy_kld_until_with_preselection__without_preselection():
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

  res = greedy_kld_uniform_ngrams_seconds_with_preselection(
    data=data,
    durations_s=durations,
    seconds=target_duration,
    preselection=preselection,
    ignore_symbols={"b"},
    duration_boundary=(0, 2.0),
    n_gram=1,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([5, 7]) == res


def test_sort_greedy_kld_until_with_preselection__with_preselection():
  preselection = OrderedDict({
    5: ["a", "b"],
  })

  data = OrderedDict({
    6: ["b"],
    7: ["c"],
  })

  durations = {
    6: 1.0,
    7: 1.0,
  }

  target_duration = 1.0

  res = greedy_kld_uniform_ngrams_seconds_with_preselection(
    data=data,
    durations_s=durations,
    seconds=target_duration,
    preselection=preselection,
    ignore_symbols={"b"},
    n_gram=1,
    duration_boundary=(0, 2.0),
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([7]) == res


def test_sort_greedy_kld_until_with_preselection__with_preselection_and_one_utterance_without_any_relevant_symbols():
  preselection = OrderedDict({
    5: ["a", "b"],
  })

  data = OrderedDict({
    6: ["b"],
    7: ["c"],
    8: ["d"],
  })

  durations = {
    6: 1.0,
    7: 1.0,
    8: 1.0,
  }

  target_duration = 2.0

  res = greedy_kld_uniform_ngrams_seconds_with_preselection(
    data=data,
    durations_s=durations,
    seconds=target_duration,
    preselection=preselection,
    ignore_symbols={"b", "d"},
    n_gram=1,
    duration_boundary=(0, 2.0),
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([7, 6]) == res


def test_sort_greedy_kld_until_with_preselection__only_ignored_symbols():
  preselection = OrderedDict({
    5: ["a", "b"],
  })

  data = OrderedDict({
    6: ["b"],
    7: ["b"],
  })

  durations = {
    6: 1.0,
    7: 1.0,
  }

  target_duration = 1.0

  res = greedy_kld_uniform_ngrams_seconds_with_preselection(
    data=data,
    durations_s=durations,
    seconds=target_duration,
    preselection=preselection,
    ignore_symbols={"b"},
    n_gram=1,
    duration_boundary=(0, 2.0),
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([6]) == res


def test_greedy_kld_uniform_ngrams_parts():
  data = OrderedDict({
    1: ["a"],  # part 1
    2: ["a"],  # part 1
    3: ["b"],  # part 2
    4: ["b"],  # part 2
  })

  res = greedy_kld_uniform_ngrams_parts(
    data=data,
    n_gram=1,
    ignore_symbols={},
    take_per_part=1,
    parts_count=2,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([1, 3, 2, 4]) == res
