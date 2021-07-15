from enum import IntEnum
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import TypeVar

from ordered_set import OrderedSet

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


class SelectionMode(IntEnum):
  FIRST = 0
  LAST = 1
  SHORTEST = 2
  LONGEST = 3


def select_key(potential_keys: OrderedSet[_T1], data: OrderedDictType[_T1, List[_T2]], mode: SelectionMode):
  assert len(potential_keys) > 0
  if len(potential_keys) == 1:
    return potential_keys[0]
  if mode == SelectionMode.FIRST:
    return select_first(potential_keys)
  if mode == SelectionMode.LAST:
    return select_last(potential_keys)
  if mode == SelectionMode.SHORTEST:
    return select_shortest(potential_keys, data)
  if mode == SelectionMode.LONGEST:
    return select_longest(potential_keys, data)
  raise Exception()


def select_first(potential_keys: OrderedSet[_T1]) -> _T1:
  assert len(potential_keys) > 0
  first = potential_keys[0]
  return first


def select_last(potential_keys: OrderedSet[_T1]) -> _T1:
  assert len(potential_keys) > 0
  first = potential_keys[-1]
  return first


def select_longest(potential_keys: OrderedSet[_T1], data: OrderedDictType[_T1, List[_T2]]) -> _T1:
  total_counts = []
  for potential_key in potential_keys:
    assert potential_key in data
    symbols = data[potential_key]
    utterance_len = len(symbols)
    total_counts.append((potential_key, utterance_len))
  key, _ = max(total_counts, key=lambda kc: kc[1])
  return key


def select_shortest(potential_keys: OrderedSet[_T1], data: OrderedDictType[_T1, List[_T2]]) -> _T1:
  total_counts = []
  for potential_key in potential_keys:
    assert potential_key in data
    symbols = data[potential_key]
    utterance_len = len(symbols)
    total_counts.append((potential_key, utterance_len))
  key, _ = min(total_counts, key=lambda kc: kc[1])
  return key
