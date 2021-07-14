from collections import Counter, OrderedDict
from logging import getLogger
from typing import Dict, List
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple, TypeVar, Union

import numpy as np
from ordered_set import OrderedSet
from scipy.stats import entropy
from tqdm import tqdm, trange

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def sort_greedy_kld(data: OrderedDictType[_T1, List[_T2]], target_dist: Dict[_T2, float]) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  logger = getLogger(__name__)
  result: OrderedSet[_T1] = OrderedSet()
  all_keys: Set[_T2] = set(target_dist.keys())
  all_occuring_values: Set[_T2] = {x for y in data.values() for x in y}
  assert all_keys == all_occuring_values

  logger.info("Preparing data...")
  covered_array = dict_to_array_ordered_after_keys({x: 0 for x in all_keys})
  target_dist_array = dict_to_array_ordered_after_keys(target_dist)
  available_entries_array = get_available_arrays(data, all_keys)

  logger.info("Selecting data...")
  for _ in trange(len(available_entries_array)):
    selected_key, selected_counter = get_utterance_with_min_kld(
      data=available_entries_array,
      covered_counts=covered_array,
      target_dist=target_dist_array
    )
    result.add(selected_key)
    available_entries_array.pop(selected_key)
    covered_array += selected_counter
  return result


def sort_greedy_kld_iterations(data: OrderedDictType[_T1, List[_T2]], target_dist: Dict[_T1, float], iterations: int) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  logger = getLogger(__name__)
  result: OrderedSet[_T1] = OrderedSet()
  all_keys = set(target_dist.keys())
  all_occuring_values: Set[_T2] = {x for y in data.values() for x in y}
  assert all_keys == all_occuring_values

  logger.info("Preparing data...")
  covered_array = dict_to_array_ordered_after_keys({x: 0 for x in all_keys})
  target_dist_array = dict_to_array_ordered_after_keys(target_dist)
  available_entries_array = get_available_arrays(data, all_keys)

  logger.info("Selecting data...")
  its = min(iterations, len(available_entries_array))
  for _ in trange(its):
    selected_key, selected_counts = get_utterance_with_min_kld(
      data=available_entries_array,
      covered_counts=covered_array,
      target_dist=target_dist_array
    )
    result.add(selected_key)
    available_entries_array.pop(selected_key)
    covered_array += selected_counts
  return result


def sort_greedy_kld_until(data: OrderedDictType[_T1, List[_T2]], target_dist: Dict[_T1, float], until_values: Dict[_T1, Union[float, int]], until_value: Union[float, int]) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  logger = getLogger(__name__)
  result: OrderedSet[_T1] = OrderedSet()
  all_keys = set(target_dist.keys())
  all_occuring_values: Set[_T2] = {x for y in data.values() for x in y}
  assert all_keys == all_occuring_values

  logger.info("Preparing data...")
  covered_array = dict_to_array_ordered_after_keys({x: 0 for x in all_keys})
  target_dist_array = dict_to_array_ordered_after_keys(target_dist)
  available_entries_array = get_available_arrays(data, all_keys)

  logger.info("Selecting data...")
  max_until = sum(until_values.values())
  adjusted_until = round(min(until_value, max_until))
  current_total = 0
  progress_bar = tqdm(total=adjusted_until, initial=current_total)
  while len(available_entries_array) > 0:
    selected_key, selected_counts = get_utterance_with_min_kld(
      data=available_entries_array,
      covered_counts=covered_array,
      target_dist=target_dist_array
    )
    selected_until_value = until_values[selected_key]
    new_total = current_total + selected_until_value
    if new_total <= until_value:
      result.add(selected_key)
      available_entries_array.pop(selected_key)
      covered_array += selected_counts
      current_total = new_total
      progress_bar.update(round(selected_until_value))
    else:
      break
  progress_bar.close()
  return result


def sort_greedy_kld_until_with_preselection(data: OrderedDictType[_T1, List[_T2]], target_dist: Dict[_T1, float], until_values: Dict[_T1, Union[float, int]], until_value: Union[float, int], preselection: OrderedDictType[_T1, List[_T2]]) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  logger = getLogger(__name__)
  result: OrderedSet[_T1] = OrderedSet()
  all_keys = set(target_dist.keys())
  all_occuring_values: Set[_T2] = {x for y in data.values() for x in y}
  assert all_keys == all_occuring_values

  logger.info("Preparing data...")
  covered_array = dict_to_array_ordered_after_keys({x: 0 for x in all_keys})
  target_dist_array = dict_to_array_ordered_after_keys(target_dist)
  available_entries_array = get_available_arrays(data, all_keys)

  logger.info("Selecting data...")
  max_until = sum(until_values.values())
  adjusted_until = round(min(until_value, max_until))
  current_total = 0
  progress_bar = tqdm(total=adjusted_until, initial=current_total)
  while len(available_entries_array) > 0:
    selected_key, selected_counts = get_utterance_with_min_kld(
      data=available_entries_array,
      covered_counts=covered_array,
      target_dist=target_dist_array
    )
    selected_until_value = until_values[selected_key]
    new_total = current_total + selected_until_value
    if new_total <= until_value:
      result.add(selected_key)
      available_entries_array.pop(selected_key)
      covered_array += selected_counts
      current_total = new_total
      progress_bar.update(round(selected_until_value))
    else:
      break
  progress_bar.close()
  return result


def get_utterance_with_min_kld(data: OrderedDictType[_T1, np.ndarray], covered_counts: np.ndarray, target_dist: Dict[_T1, float]) -> Tuple[_T1, np.ndarray]:
  assert isinstance(data, OrderedDict)
  divergences = {k: get_divergence_for_utterance(
      covered_counts=covered_counts,
      utterance_counts=utterance_counts,
      target_dist=target_dist,
    ) for k, utterance_counts in data.items()
  }

  selected_key, selected_divergence = min(divergences.items(), key=lambda kv: kv[1])
  selected_counter = data[selected_key]
  return selected_key, selected_counter


def get_divergence_for_utterance(covered_counts: np.ndarray, utterance_counts: np.ndarray, target_dist: Dict[_T1, float]) -> float:
  counts = covered_counts + utterance_counts
  distr = _get_distribution(counts)
  res = entropy(distr, target_dist)
  return res


def get_available_arrays(data: OrderedDictType[_T1, List[_T2]], all_keys: Set[_T1]) -> OrderedDictType[_T1, np.ndarray]:
  assert isinstance(data, OrderedDict)
  available_entries_counter: OrderedDictType[_T1, Counter] = OrderedDict({
    k: Counter(v) for k, v in data.items()
  })

  for k in available_entries_counter:
    add_missing_keys(available_entries_counter[k], all_keys)

  available_entries_array: OrderedDictType[_T1, np.ndarray] = OrderedDict({
    k: dict_to_array_ordered_after_keys(
        counter) for k, counter in available_entries_counter.items()
  })

  return available_entries_array


def _get_distribution(counts: np.ndarray) -> np.ndarray:
  new_dist = np.divide(counts, np.sum(counts))
  return new_dist


def get_uniform_distribution(ngrams: Dict[_T1, List[_T2]]) -> Dict[_T2, float]:
  unique_ngrams: Set[_T2] = {x for y in ngrams.values() for x in y}
  distr = 1 / len(unique_ngrams)
  res: Dict[_T2, float] = {k: distr for k in unique_ngrams}
  return res


def add_missing_keys(counter: Counter, keys: Set[_T1]) -> None:
  for k in keys:
    if k not in counter:
      counter[k] = 0


def dict_to_array_ordered_after_keys(d: Dict) -> np.ndarray:
  res = np.array([d[k] for k in sorted(d.keys())])
  return res
