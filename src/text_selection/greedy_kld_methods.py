import math
from collections import Counter, OrderedDict
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from logging import getLogger
from multiprocessing import cpu_count
from typing import Any, Dict, List
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple, TypeVar, Union

import numpy as np
from ordered_set import OrderedSet
from scipy.stats import entropy
from tqdm import tqdm, trange

from text_selection.selection import SelectionMode, select_first, select_key

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def split_into_equal_parts(keys: OrderedSet[_T1], parts_count: int) -> List[OrderedSet[_T1]]:
  assert parts_count > 0
  if len(keys) == 0:
    return []

  chunk_size = math.ceil(len(keys) / parts_count)

  result = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]

  return result


# def split_into_equal_parts_diverse_lengths_per_set(keys: OrderedSet[_T1], parts_count: int) -> List[OrderedSet[_T1]]:
#   assert parts_count > 0
#   tmp: OrderedDictType[int, OrderedSet[_T1]] = OrderedDict({
#     part_id: OrderedSet() for part_id in range(parts_count)
#   })

#   current_part_id = 0
#   for key in keys:
#     tmp[current_part_id].add(key)
#     current_part_id += 1
#     if current_part_id == parts_count:
#       current_part_id = 0
#   result = list(tmp.values())
#   return result


def get_keys_sort_after_value(data: OrderedDictType[_T1, Union[int, float]]) -> OrderedSet[_T1]:
  # required to have predictable results when equal entries exist in values
  assert isinstance(data, OrderedDict)
  if len(data) == 0:
    return OrderedSet()
  sorted_keys, _ = list(zip(*sorted(data.items(), key=lambda kv: kv[1])))
  result = OrderedSet(sorted_keys)
  return result


# def apply_sorting(sorted_keys: OrderedSet[_T1], data: Dict[_T1, _T2]) -> OrderedDictType[_T1, _T2]:
#   assert set(sorted_keys) == set(data.keys())
#   result = OrderedDict({k: data[k] for k in sorted_keys})
#   return result


def sort_kld_parts(data: Dict[_T1, List[_T2]], target_dist: Dict[_T2, float], parts_count: int, take_per_part: int, lengths: OrderedDictType[_T1, Union[int, float]]) -> OrderedSet[_T1]:
  logger = getLogger(__name__)

  selection_mode = SelectionMode.FIRST
  result: OrderedSet[_T1] = OrderedSet()
  all_keys_in_targed_distr = set(target_dist.keys())

  logger.info("Preparing data...")
  target_dist_array = dict_to_array_ordered_after_keys(target_dist)
  covered_array = dict_to_array_ordered_after_keys({x: 0 for x in all_keys_in_targed_distr})

  sorted_keys = get_keys_sort_after_value(lengths)
  parts = split_into_equal_parts(sorted_keys, parts_count)

  parts_data: List[np.ndarray] = []
  for part in parts:
    data_part = OrderedDict({k: data[k] for k in part})
    data_arrays = get_available_arrays(data_part, all_keys_in_targed_distr)
    parts_data.append(data_arrays)

  logger.info("Selecting data...")
  progress_bar = tqdm(total=len(data))

  while len(result) != len(data):
    for part in parts_data:
      available_entries_array = part
      for _ in range(take_per_part):
        if len(available_entries_array) == 0:
          continue
        potential_keys = get_utterance_with_min_kld(
          data=available_entries_array,
          covered_counts=covered_array,
          target_dist=target_dist_array,
          mp=False,
        )

        if len(potential_keys) > 1:
          logger.info(f"Found {len(potential_keys)} candidates for the current iteration.")
        selected_key = select_key(potential_keys, unit_counts=None, mode=selection_mode)

        result.add(selected_key)
        covered_array += available_entries_array[selected_key]
        available_entries_array.pop(selected_key)
        progress_bar.update()
  progress_bar.close()

  final_distr = _get_distribution(covered_array)
  final_kld = entropy(final_distr, target_dist_array)
  logger.info(f"Obtained Kullback-Leibler divergence: {final_kld}")

  return result


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
    potential_keys = get_utterance_with_min_kld(
      data=available_entries_array,
      covered_counts=covered_array,
      target_dist=target_dist_array,
      mp=False,
    )
    selected_key = select_first(potential_keys)
    result.add(selected_key)
    covered_array += available_entries_array[selected_key]
    available_entries_array.pop(selected_key)
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
    potential_keys = get_utterance_with_min_kld(
      data=available_entries_array,
      covered_counts=covered_array,
      target_dist=target_dist_array,
      mp=False,
    )
    selected_key = select_first(potential_keys)
    result.add(selected_key)
    covered_array += available_entries_array[selected_key]
    available_entries_array.pop(selected_key)
  return result


def sort_greedy_kld_until(data: OrderedDictType[_T1, List[_T2]], target_dist: Dict[_T1, float], until_values: Dict[_T1, Union[float, int]], until_value: Union[float, int]) -> OrderedSet[_T1]:
  selection = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=target_dist,
    until_values=until_values,
    until_value=until_value,
    preselection=OrderedDict(),
    mp=False,
  )

  return selection


def sort_greedy_kld_until_with_preselection(data: OrderedDictType[_T1, List[_T2]], target_dist: Dict[_T2, float], until_values: Dict[_T1, Union[float, int]], until_value: Union[float, int], preselection: OrderedDictType[_T1, List[_T2]], mp: bool) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  assert isinstance(preselection, OrderedDict)
  # The probability is really high that only one key is figured out, therefore it is useless to use any selection modes. If shortest or longest should be used the unfiltered count of symbols needs to be passed as extra parameter which increases complexity of the method.
  selection_mode = SelectionMode.FIRST
  logger = getLogger(__name__)
  if mp:
    logger.info("Using multiprocessing...")
  result: OrderedSet[_T1] = OrderedSet()
  all_keys_in_targed_distr = set(target_dist.keys())
  # all_occuring_values: Set[_T2] = {x for y in data.values() for x in y}
  # assert all_keys == all_occuring_values

  logger.info("Preparing data...")
  target_dist_array = dict_to_array_ordered_after_keys(target_dist)
  available_entries_array = get_available_arrays(data, all_keys_in_targed_distr)

  if len(preselection) == 0:
    covered_array = dict_to_array_ordered_after_keys({x: 0 for x in all_keys_in_targed_distr})
  else:
    logger.info("Using preselected data.")
    preselection_array = get_available_arrays(preselection, all_keys_in_targed_distr)
    covered_array = merge_arrays(preselection_array)

    preselection_distr = _get_distribution(covered_array)
    preselection_kld = entropy(preselection_distr, target_dist_array)
    logger.info(f"Preselection Kullback-Leibler divergence: {preselection_kld}")

  all_occuring_ngrams_data: Set[_T2] = {x for y in data.values() for x in y}
  all_occuring_ngrams_preselection: Set[_T2] = {x for y in preselection.values() for x in y}
  all_occuring_ngrams: Set[_T2] = all_occuring_ngrams_data | all_occuring_ngrams_preselection
  missing_keys_in_data_and_preselection = all_keys_in_targed_distr - all_occuring_ngrams
  if len(missing_keys_in_data_and_preselection) > 0:
    # is only relevant if the targed_distr for these keys is not zero.
    logger.warning(
      f"Some keys from targed distribution do not exist in data and preselection: {'.'.join(sorted(missing_keys_in_data_and_preselection))}")

  logger.info("Selecting data...")
  max_until = sum(until_values.values())
  adjusted_until = round(min(until_value, max_until))
  current_total = 0.0
  progress_bar = tqdm(total=adjusted_until, initial=round(current_total))
  while True:
    if len(available_entries_array) == 0:
      logger.warning(
        f"Aborting selection as no further data is available! Selected: {current_total:.1f}/{until_value:.1f} ({current_total/until_value*100:.2f}%).")
      break
    potential_keys = get_utterance_with_min_kld(
      data=available_entries_array,
      covered_counts=covered_array,
      target_dist=target_dist_array,
      mp=mp,
    )
    if len(potential_keys) > 1:
      logger.info(f"Found {len(potential_keys)} candidates for the current iteration.")
    selected_key = select_key(potential_keys, unit_counts=None, mode=selection_mode)
    selected_until_value = until_values[selected_key]
    new_total = current_total + selected_until_value
    if new_total <= until_value:
      result.add(selected_key)
      covered_array += available_entries_array[selected_key]
      current_total = new_total
      available_entries_array.pop(selected_key)
      progress_bar.update(round(selected_until_value))
      if current_total == until_value:
        break
    else:
      break
  progress_bar.close()

  final_distr = _get_distribution(covered_array)
  final_kld = entropy(final_distr, target_dist_array)
  logger.info(f"Obtained Kullback-Leibler divergence: {final_kld}")

  return result


def get_utterance_with_min_kld(data: OrderedDictType[_T1, np.ndarray], covered_counts: np.ndarray, target_dist: np.ndarray, mp: bool) -> OrderedSet[_T1]:
  get_divergences_method = get_divergences_mp if mp else get_divergences
  divergences = get_divergences_method(data, covered_counts, target_dist)
  all_with_minimum_divergence = get_smallest_divergence_keys(divergences)
  return all_with_minimum_divergence


def get_divergences(data: OrderedDictType[_T1, np.ndarray], covered_counts: np.ndarray, target_dist: np.ndarray) -> OrderedDictType[_T1, float]:
  assert isinstance(data, OrderedDict)
  divergences = OrderedDict({k: get_divergence_for_utterance(
      covered_counts=covered_counts,
      utterance_counts=utterance_counts,
      target_dist=target_dist,
    ) for k, utterance_counts in data.items()
  })

  return divergences


def get_divergences_mp(data: OrderedDictType[_T1, np.ndarray], covered_counts: np.ndarray, target_dist: np.ndarray) -> OrderedDictType[_T1, float]:
  assert isinstance(data, OrderedDict)

  logger = getLogger(__name__)
  thread_count = cpu_count() - 1
  chunksize = math.ceil(len(data) / thread_count)
  logger.debug(f"Using {thread_count} threads with {chunksize} chunks...")

  meth = partial(get_divergence_for_utterance_mp,
                 covered_counts=covered_counts, target_dist=target_dist)
  # res = process_map(meth, data.items(), max_workers=thread_count, chunksize=chunksize) # is equivalent but a bit slower
  with ProcessPoolExecutor(max_workers=thread_count) as ex:
    #res = dict(tqdm(ex.map(meth, data.items(), chunksize=chunksize), total=len(data)))
    res = dict(ex.map(meth, data.items(), chunksize=chunksize))

  result: OrderedDictType[_T1, float] = OrderedDict({k: res[k] for k in data})

  return result


def get_smallest_divergence_keys(divergences: OrderedDictType[_T1, float]) -> OrderedSet[_T1]:
  assert isinstance(divergences, OrderedDict)
  assert len(divergences) > 0
  #selected_key, minimum_divergence = min(divergences.items(), key=lambda kv: kv[1])
  minimum_divergence = min(divergences.values())
  all_with_minimum_divergence = OrderedSet([key for key, divergence in divergences.items()
                                            if divergence == minimum_divergence])
  return all_with_minimum_divergence


def get_divergence_for_utterance_mp(kv_pair: Tuple[_T1, np.ndarray], covered_counts: np.ndarray, target_dist: np.ndarray) -> Tuple[_T1, float]:
  key, utterance_counts = kv_pair
  return key, get_divergence_for_utterance(covered_counts, utterance_counts, target_dist)


def get_divergence_for_utterance(covered_counts: np.ndarray, utterance_counts: np.ndarray, target_dist: np.ndarray) -> float:
  counts = covered_counts + utterance_counts
  distr = _get_distribution(counts)
  res = get_kld(distr, target_dist)
  return res


def get_kld(dist: np.ndarray, target_dist: np.ndarray) -> Tuple[_T1, float]:
  none_of_targed_ngrams_exist = all(np.isnan(dist))
  if none_of_targed_ngrams_exist:
    return math.inf

  res = entropy(dist, target_dist)
  return res


def get_available_arrays(data: OrderedDictType[_T1, List[_T2]], all_keys: Set[_T1]) -> OrderedDictType[_T1, np.ndarray]:
  assert isinstance(data, OrderedDict)
  available_entries_counter: OrderedDictType[_T1, Counter] = OrderedDict({
    k: Counter(v) for k, v in data.items()
  })

  for k in available_entries_counter:
    sync_dict_keys_to_keys(available_entries_counter[k], all_keys)

  available_entries_array: OrderedDictType[_T1, np.ndarray] = OrderedDict({
    k: dict_to_array_ordered_after_keys(
        counter) for k, counter in available_entries_counter.items()
  })

  return available_entries_array


def merge_arrays(data: Dict[_T1, np.ndarray]) -> np.ndarray:
  assert len(data) > 0
  merged_array = None
  for array in data.values():
    if merged_array is None:
      merged_array = array
    else:
      merged_array += array
  return merged_array


def _get_distribution(counts: np.ndarray) -> np.ndarray:
  # assert all(x >= 0 for x in counts) # too slow
  sum_counts = np.sum(counts)
  new_dist = np.divide(counts, sum_counts)
  return new_dist

# def _get_distribution(counts: np.ndarray) -> np.ndarray:
#   assert all(x >= 0 for x in counts)

#   sum_counts = np.sum(counts)

#   if sum_counts == 0:
#     return counts
#   else:
#     new_dist = np.divide(counts, sum_counts)
#     return new_dist


def get_uniform_distribution(ngrams: Dict[_T1, List[_T2]]) -> Dict[_T2, float]:
  unique_ngrams: Set[_T2] = {x for y in ngrams.values() for x in y}
  if len(unique_ngrams) == 0:
    return dict()
  distr = 1 / len(unique_ngrams)
  res: Dict[_T2, float] = {k: distr for k in unique_ngrams}
  return res


def sync_dict_keys_to_keys(counter: Dict[_T1, int], keys: Set[_T1]) -> None:
  for k in keys:
    if k not in counter:
      counter[k] = 0
  keys_to_remove = set()
  for existing_key in counter:
    if existing_key not in keys:
      keys_to_remove.add(existing_key)
  for key_to_remove in keys_to_remove:
    counter.pop(key_to_remove)


def dict_to_array_ordered_after_keys(d: Dict) -> np.ndarray:
  res = np.array([d[k] for k in sorted(d.keys())])
  return res
