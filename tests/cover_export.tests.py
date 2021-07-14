from text_selection.cover_export import *
from text_selection.utils import *


def test_cover_symbols_default():
  data = OrderedDict({
    1: ["a", "a"],  # one new
    2: ["c", "a"],  # two new
    3: ["d", "a"],  # two new
  })

  res = cover_symbols_default(
    data=data,
    symbols={"c"},
  )

  assert OrderedSet([2]) == res


def test_cover_symbols_default__not_existing_symbols__are_ignored():
  data = OrderedDict({
    1: ["a", "a"],  # one new
    2: ["c", "a"],  # two new
    3: ["d", "a"],  # two new
  })

  res = cover_symbols_default(
    data=data,
    symbols={"x"},
  )

  assert OrderedSet([]) == res
