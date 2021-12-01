from text_selection.greedy.greedy_export import *
from text_selection.utils import *


def test_greedy_ngrams_cover__one_grams__return_correct_sorting():
  data = OrderedDict({
    1: ["a", "a", "a"],
    2: ["c", "c"],
    3: ["d"],
    4: ["e"],
  })

  res = greedy_ngrams_cover(
    data=data,
    already_covered=OrderedDict(),
    top_percent=0.5,
    n_gram=1,
    ignore_symbols=None,
  )

  assert_res = OrderedSet([1, 2])
  assert assert_res == res


def test_greedy_ngrams_cover__one_grams__return_correct_sorting_after_value():
  data = OrderedDict({
    1: ["a"],
    2: ["c"],
    3: ["b"],
    4: ["d"],
  })

  res = greedy_ngrams_cover(
    data=data,
    already_covered=OrderedDict(),
    top_percent=0.5,
    n_gram=1,
    ignore_symbols=None,
  )

  assert_res = OrderedSet([1, 3])
  assert assert_res == res


def test_greedy_ngrams_cover__one_grams_and_already_covered__return_correct_sorting():
  data = OrderedDict({
    2: ["c", "c"],
    3: ["d"],
    4: ["e"],
  })

  already_covered = OrderedDict({
    1: ["a", "a", "a"],  # one new
  })

  res = greedy_ngrams_cover(
    data=data,
    already_covered=already_covered,
    top_percent=0.5,
    n_gram=1,
    ignore_symbols=None,
  )

  assert_res = OrderedSet([2])
  assert assert_res == res


def test_greedy_ngrams_cover__one_grams_and_already_covered_and_ignoring__return_correct_sorting():
  data = OrderedDict({
    2: ["c", "c", "c"],
    3: ["d"],
    4: ["e", "e"],
    5: ["f"],
  })

  already_covered = OrderedDict({
    1: ["a", "a", "a", "a"],  # one new
  })

  res = greedy_ngrams_cover(
    data=data,
    already_covered=already_covered,
    top_percent=0.5,
    n_gram=1,
    ignore_symbols={"a"},
  )

  assert_res = OrderedSet([2, 4])
  assert assert_res == res


def test_greedy_ngrams_iterations__one_grams__return_correct_sorting():
  data = OrderedDict({
    1: ["a", "a"],  # one new
    2: ["c", "a"],  # two new
    3: ["d", "a"],  # two new
  })

  res = greedy_ngrams_iterations(
    data=data,
    n_gram=1,
    ignore_symbols=None,
    iterations=3,
  )

  assert_res = OrderedSet([2, 3, 1])
  assert assert_res == res


def test_greedy_ngrams_epochs__one_grams_one_epoch__return_correct_sorting():
  data = OrderedDict({
    1: ["a", "a"],  # one new
    2: ["c", "a"],  # two new
    3: ["d", "a"],  # two new
  })

  res = greedy_ngrams_epochs(
    data=data,
    n_gram=1,
    ignore_symbols=None,
    epochs=1,
  )

  assert_res = OrderedSet([2, 3])
  assert assert_res == res


def test_greedy_ngrams_seconds__one_gram_two_seconds__return_correct_sorting():
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

  res = greedy_ngrams_seconds(
    data=data,
    n_gram=1,
    ignore_symbols=None,
    durations_s=durations,
    seconds=2,
  )

  assert_res = OrderedSet([2, 3])
  assert assert_res == res


def test_greedy_ngrams_cound__one_gram_two_counts__return_correct_sorting():
  data = OrderedDict({
    1: ["a", "a"],  # one new
    2: ["c", "a"],  # two new
    3: ["d", "a"],  # two new
  })

  counts = {
    1: 1,
    2: 1,
    3: 1,
  }

  res = greedy_ngrams_count(
    data=data,
    n_gram=1,
    ignore_symbols=None,
    chars=counts,
    total_count=2,
  )

  assert_res = OrderedSet([2, 3])
  assert assert_res == res
