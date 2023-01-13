
from collections import OrderedDict

from ordered_set import OrderedSet

from text_selection.selection import (SelectionMode, select_first, select_key, select_last,
                                      select_longest, select_shortest)


def test_select_longest__selects_longest():
  potential_keys = OrderedSet([0, 1, 2])
  data = OrderedDict({
    0: [1, 1],
    1: [1, 1, 1],
    2: [1],
  })

  result = select_longest(potential_keys, data)

  assert result == 1


def test_select_longest__two_candidates_selects_first():
  potential_keys = OrderedSet([0, 1, 2])
  data = OrderedDict({
    0: [1, 1],
    1: [1, 1, 1],
    2: [1, 1],
    3: [1, 1, 1],
    4: [1, 1],
  })

  result = select_longest(potential_keys, data)

  assert result == 1


def test_select_shortest__selects_shortest():
  potential_keys = OrderedSet([0, 1, 2])
  data = OrderedDict({
    0: [1, 1],
    1: [1, 1, 1],
    2: [1],
  })

  result = select_shortest(potential_keys, data)

  assert result == 2


def test_select_shortest__two_candidates_selects_first():
  potential_keys = OrderedSet([0, 1, 2])
  data = OrderedDict({
    0: [1, 1, 1],
    1: [1, 1],
    2: [1, 1, 1],
    3: [1, 1],
    4: [1, 1, 1],
  })

  result = select_shortest(potential_keys, data)

  assert result == 1


def test_select_first__selects_first():
  potential_keys = OrderedSet([0, 1, 2])

  result = select_first(potential_keys)

  assert result == 0


def test_select_last__selects_last():
  potential_keys = OrderedSet([0, 1, 2])

  result = select_last(potential_keys)

  assert result == 2


def test_select_key__one_entry__returns_it():
  potential_keys = OrderedSet([1])
  data = OrderedDict({
    1: [1, 1, 1],
  })

  key_first = select_key(potential_keys, data, mode=SelectionMode.FIRST)
  key_last = select_key(potential_keys, data, mode=SelectionMode.LAST)
  key_shortest = select_key(potential_keys, data, mode=SelectionMode.SHORTEST)
  key_longest = select_key(potential_keys, data, mode=SelectionMode.LONGEST)

  assert key_first == key_last == key_shortest == key_longest == 1


def test_select_key__first():
  potential_keys = OrderedSet([0, 1, 2, 3])
  data = OrderedDict({
    0: [1, 1, 1],
    1: [1, 1],
    2: [1, 1, 1, 1],
    3: [1, 1, 1],
  })

  key_first = select_key(potential_keys, data, mode=SelectionMode.FIRST)

  assert key_first == 0


def test_select_key__last():
  potential_keys = OrderedSet([0, 1, 2, 3])
  data = OrderedDict({
    0: [1, 1, 1],
    1: [1, 1],
    2: [1, 1, 1, 1],
    3: [1, 1, 1],
  })

  key_first = select_key(potential_keys, data, mode=SelectionMode.LAST)

  assert key_first == 3


def test_select_key__shortest():
  potential_keys = OrderedSet([0, 1, 2, 3])
  data = OrderedDict({
    0: [1, 1, 1],
    1: [1, 1],
    2: [1, 1, 1, 1],
    3: [1, 1, 1],
  })

  key_first = select_key(potential_keys, data, mode=SelectionMode.SHORTEST)

  assert key_first == 1


def test_select_key__longest():
  potential_keys = OrderedSet([0, 1, 2, 3])
  data = OrderedDict({
    0: [1, 1, 1],
    1: [1, 1],
    2: [1, 1, 1, 1],
    3: [1, 1, 1],
  })

  key_first = select_key(potential_keys, data, mode=SelectionMode.LONGEST)

  assert key_first == 2
