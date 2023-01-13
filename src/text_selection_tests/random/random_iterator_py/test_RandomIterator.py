from ordered_set import OrderedSet

from text_selection.random.random_iterator import RandomIterator


def test_empty__returns_empty_list():
  keys = OrderedSet()
  iterator = RandomIterator(keys, seed=0)

  result = list(iterator)

  assert result == []


def test_component_seed_1():
  keys = OrderedSet(range(5))
  iterator = RandomIterator(keys, seed=0)

  result = list(iterator)

  assert result == [3, 4, 0, 2, 1]


def test_component_seed_0():
  keys = OrderedSet(range(5))
  iterator = RandomIterator(keys, seed=1)

  result = list(iterator)

  assert result == [1, 0, 3, 2, 4]
