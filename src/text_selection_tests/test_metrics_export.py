import math
from collections import OrderedDict

from text_selection.metrics_export import get_rarity_ngrams


def test_get_rarity_ngrams__one_entry():
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

  assert assert_res == res


def test_get_rarity_ngrams__two_entries():
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

  assert assert_res == res


def test_get_rarity_ngrams__not_existing__has_zero():
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

  assert assert_res == res


def test_get_rarity_ngrams__one_ignored_in_data__ignores_it():
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

  assert assert_res == res


def test_get_rarity_ngrams__one_ignored_in_corpus__ignores_it():
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

  assert assert_res == res


def test_get_rarity_ngrams__empty_entry__returns_inf():
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

  assert assert_res == res
