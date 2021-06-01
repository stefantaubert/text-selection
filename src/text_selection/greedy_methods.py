from collections import OrderedDict
from typing import Dict
from typing import OrderedDict as OrderedDictType
from typing import Set, TypeVar, Union

from ordered_set import OrderedSet
from tqdm import tqdm

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def get_greedy_cover(data: OrderedDictType[_T1, Set[_T2]], already_covered: Set[_T2]) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  data_containing_only_new_units = OrderedDict(
    {k: v for k, v in data.items() if len(v.difference(already_covered)) > 0}
  )
  all_already_covered = len(data_containing_only_new_units) == 0

  if all_already_covered:
    return OrderedSet()

  res = get_greedy(data_containing_only_new_units)
  return res


def sort_greedy(data: OrderedDictType[_T1, Set[_T2]]) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  result: OrderedSet[_T1] = OrderedSet()
  available_entries = data.copy()
  progress_bar = tqdm(total=len(data), initial=0)
  while len(available_entries) > 0:
    selection = get_greedy(available_entries)
    result.update(selection)
    for k in selection:
      available_entries.pop(k)
    progress_bar.update(round(len(result) - progress_bar.n, 0))
  progress_bar.close()
  return result


def sort_greedy_epochs(data: OrderedDictType[_T1, Set[_T2]], epochs: int) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  assert epochs >= 0
  result: OrderedSet[_T1] = OrderedSet()
  available_entries = data.copy()
  epochs_done = 0
  epochs_goal = min(epochs, len(available_entries))
  progress_bar = tqdm(total=epochs_goal, initial=0)
  while len(available_entries) > 0 and epochs_done != epochs_goal:
    selection = get_greedy(available_entries)
    result.update(selection)
    for selected_key in selection:
      available_entries.pop(selected_key)
    epochs_done += 1
    progress_bar.update(1)
  progress_bar.close()
  return result


def sort_greedy_until(data: OrderedDictType[_T1, Set[_T2]], until_values: Dict[_T1, Union[float, int]], until_value: Union[float, int]) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  result: OrderedSet[_T1] = OrderedSet()
  available_entries = data.copy()
  total = 0
  continue_while = True
  progress_bar = tqdm(total=int(round(until_value, 0)), initial=0)
  while continue_while and len(available_entries) > 0:
    selection = get_greedy(available_entries)
    for selected_key in selection:
      new_total = total + until_values[selected_key]
      if new_total <= until_value:
        result.add(selected_key)
        available_entries.pop(selected_key)
        total = new_total
        progress_bar.update(int(round(total - progress_bar.n, 0)))
      else:
        continue_while = False
        break
  progress_bar.close()
  return result


def get_greedy(data: OrderedDictType[_T1, Set[_T2]]) -> OrderedSet[_T1]:
  """The parameter ngrams needs to be ordered to be able to produce reproductable results."""
  assert isinstance(data, OrderedDict)
  all_ngrams = {e for s in data.values() for e in s}
  available_entries = data.copy()
  covered: Set[_T2] = set()
  result: OrderedSet[_T1] = OrderedSet()

  while covered != all_ngrams:
    selected_key, selected_value = max(
      available_entries.items(), key=lambda x: get_new_units_count(x[1], covered))
    result.add(selected_key)
    available_entries.pop(selected_key)
    covered |= selected_value

  return result


def get_new_units_count(subset: Set[_T2], already_covered: Set[_T2]) -> int:
  difference = subset - already_covered
  res = len(difference)
  return res
