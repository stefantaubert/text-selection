from enum import IntEnum
from typing import Dict, Optional, TypeVar

from ordered_set import OrderedSet

_T1 = TypeVar("_T1")


class SelectionMode(IntEnum):
  FIRST = 0
  LAST = 1
  SHORTEST = 2
  LONGEST = 3


def select_key(potential_keys: OrderedSet[_T1], data_lens: Optional[Dict[_T1, int]], mode: SelectionMode):
  assert len(potential_keys) > 0
  if len(potential_keys) == 1:
    return potential_keys[0]
  if mode == SelectionMode.FIRST:
    return select_first(potential_keys)
  if mode == SelectionMode.LAST:
    return select_last(potential_keys)
  if mode == SelectionMode.SHORTEST:
    assert data_lens is not None
    return select_shortest(potential_keys, data_lens)
  if mode == SelectionMode.LONGEST:
    assert data_lens is not None
    return select_longest(potential_keys, data_lens)
  raise Exception()


def select_first(potential_keys: OrderedSet[_T1]) -> _T1:
  assert len(potential_keys) > 0
  first = potential_keys[0]
  return first


def select_last(potential_keys: OrderedSet[_T1]) -> _T1:
  assert len(potential_keys) > 0
  first = potential_keys[-1]
  return first


def select_longest(potential_keys: OrderedSet[_T1], data: Dict[_T1, int]) -> _T1:
  total_counts = []
  for potential_key in potential_keys:
    assert potential_key in data
    utterance_len = data[potential_key]
    total_counts.append((potential_key, utterance_len))
  key, _ = max(total_counts, key=lambda key_count: key_count[1])
  return key


def select_shortest(potential_keys: OrderedSet[_T1], data: Dict[_T1, int]) -> _T1:
  total_counts = []
  for potential_key in potential_keys:
    assert potential_key in data
    utterance_len = data[potential_key]
    total_counts.append((potential_key, utterance_len))
  key, _ = min(total_counts, key=lambda key_count: key_count[1])
  return key
