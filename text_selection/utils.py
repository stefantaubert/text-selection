import math
import random
from collections import Counter, OrderedDict
from logging import getLogger
from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple, TypeVar, Union

import numpy as np
from ordered_set import OrderedSet
from sklearn.cluster import KMeans

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def get_ngrams(sentence_symbols: List[str], n: int) -> List[Tuple[str]]:
  if n < 1:
    raise Exception()

  res: List[Tuple[str]] = []
  for i in range(len(sentence_symbols) - n + 1):
    tmp = tuple(sentence_symbols[i:i + n])
    res.append(tmp)
  return res


def get_filtered_list(l: List[_T1], take_only: Set[_T1]) -> List[_T1]:
  res = [x for x in l if x in take_only]
  return res


def filter_ngrams(ngrams: List[Tuple[_T2]], ignore_symbol_ids: Set[_T1]) -> List[Tuple]:
  res = [x for x in ngrams if len(set(x).intersection(ignore_symbol_ids)) == 0]
  return res


def values_to_set(d: OrderedDictType[_T1, _T2]) -> OrderedDictType[_T1, _T2]:
  res: OrderedDictType[_T1, _T2] = OrderedDict({k: set(v) for k, v in d.items()})
  return res


def get_until_sum_set(d: OrderedSet[_T1], until_values: Dict[_T1, Union[float, int]], until_value: Union[float, int]) -> Tuple[OrderedSet[_T1], Union[float, int]]:
  total = 0
  res: OrderedSet[_T1] = OrderedSet()
  for k in d:
    current_val = until_values[k]
    include = total + current_val <= until_value
    if not include:
      break
    res.add(k)
    total += current_val

  return res, total


def find_unlike_sets(sample_set_list: List[Set[int]], n: int, seed: Optional[int]) -> Set[int]:
  sample_number = len(sample_set_list)
  if n > sample_number:
    raise ValueError(
      "The number of indices you want to choose is greater than the number of available sets.")
  if n == sample_number:
    return set(range(sample_number))
  max_id = get_max_entry(sample_set_list)
  vecs = vectorize_all_sets(sample_set_list, max_id)
  k_means = KMeans(
    n_clusters=n,
    init='k-means++',
    random_state=seed,
  )
  cluster_dists = k_means.fit_transform(vecs)
  assert cluster_dists.shape[1] == n
  cluster_labels = k_means.labels_
  first_empty_cluster_index = find_empty_clusters(cluster_labels, n)
  chosen_indices = np.argmin(cluster_dists, axis=0)
  replace_chosen_indices_that_do_not_belong_to_corresponding_cluster(
    cluster_labels, cluster_dists, chosen_indices, first_empty_cluster_index)
  unselected_indices = [index for index in range(
    sample_number) if index not in chosen_indices]
  replace_chosen_indices_that_correspond_to_empty_clusters_with_first_unused_indices(
    chosen_indices, unselected_indices, first_empty_cluster_index, n)
  assert len(chosen_indices) == n
  assert len(set(chosen_indices)) == n
  return set(chosen_indices)


def find_empty_clusters(cluster_labels: np.ndarray, n: int) -> int:
  for cluster_index in range(n):
    cluster_is_not_empty_for_this_index = any(cluster_labels == cluster_index)
    if not cluster_is_not_empty_for_this_index:
      first_empty_cluster_index = cluster_index
      return first_empty_cluster_index
  return n


def replace_chosen_indices_that_correspond_to_empty_clusters_with_first_unused_indices(chosen_indices: np.ndarray, unselected_indices: Set, first_empty_cluster_index: int, n: int) -> None:
  for cluster_index in range(first_empty_cluster_index, n):
    chosen_indices[cluster_index] = unselected_indices[0]
    unselected_indices = unselected_indices[1:]


def replace_chosen_indices_that_do_not_belong_to_corresponding_cluster(cluster_labels: np.ndarray, cluster_dists: np.ndarray, chosen_indices: List[int], first_empty_cluster_index: int) -> None:
  for cluster_index in range(first_empty_cluster_index):
    while cluster_labels[chosen_indices[cluster_index]] != cluster_index:
      cluster_dists[chosen_indices[cluster_index], cluster_index] = np.inf
      chosen_indices[cluster_index] = np.argmin(cluster_dists[:, cluster_index])


def vectorize_all_sets(sample_set_list: List[Set[int]], max_id: int) -> List[List[int]]:
  vec_list = [vectorize_set(sample_set, max_id) for sample_set in sample_set_list]
  return vec_list


def vectorize_set(sample_set: Set[int], max_id: int) -> List[int]:
  vec = [int(i in sample_set) for i in range(max_id + 1)]
  return vec


def get_max_entry(sample_set_list: List[Set[int]]) -> int:
  max_of_each_set = [max(sample_set) for sample_set in sample_set_list]
  return max(max_of_each_set)


def get_random_subset_indices(sample_set_list: List[Set[int]], n: int) -> List[Set[int]]:
  chosen_indices = random.sample(range(len(sample_set_list)), n)
  while len(set(chosen_indices)) != n:
    chosen_indices = random.sample(range(len(sample_set_list)), n)
  #chosen_sets = [sample_set_list[i] for i in range(len(sample_set_list)) if i in chosen_indices]
  return chosen_indices


def get_chosen_sets(sample_set_list: List[Set[int]], chosen_indices: Set[int]) -> List[Set[int]]:
  ordered_chosen_indices = sorted(list(chosen_indices))
  chosen_sets = [sample_set_list[index] for index in ordered_chosen_indices]
  return chosen_sets


def get_total_number_of_common_elements(chosen_sets: List[Set[int]]) -> int:
  common_elements_dict = get_number_of_common_elements(chosen_sets)
  total_number = sum(common_elements_dict.values()) / 2
  return int(total_number)


def get_number_of_common_elements(chosen_sets: List[Set[int]]) -> Dict[int, int]:
  dict_of_common_elements = {index: sum(list_of_numbers_of_common_elements_for_one_index(
    chosen_sets, index)) for index in range(len(chosen_sets))}
  return dict_of_common_elements


def get_number_of_common_elements_per_set(chosen_sets: List[Set[int]]) -> Dict[int, List[int]]:
  dict_of_common_elements = {index: list_of_numbers_of_common_elements_for_one_index(
    chosen_sets, index) for index in range(len(chosen_sets))}
  return dict_of_common_elements


def list_of_numbers_of_common_elements_for_one_index(chosen_sets: List[Set[int]], index: int) -> List[int]:
  common_number_list = [len(chosen_sets[index] & chosen_set)
                        for set_index, chosen_set in enumerate(chosen_sets) if set_index != index]
  return common_number_list


def get_top_n(data: OrderedDictType[int, List[_T1]], top_percent: float) -> OrderedSet[_T1]:
  """
  if the distribution is same then it will take the _T1 by ascending sorting to make it deterministic
  """
  distr = get_distribution(data)
  top_n = round(len(distr) * top_percent)
  distr_sorted_after_key = OrderedDict(sorted(distr.items(), key=lambda x: x[0], reverse=False))
  distr_sorted = OrderedDict(sorted(distr_sorted_after_key.items(),
                                    key=lambda x: x[1], reverse=True))
  top_ngrams: OrderedSet[_T1] = OrderedSet(list(distr_sorted.keys())[:top_n])
  return top_ngrams


def get_distribution(ngrams: Dict[_T1, List[_T2]]) -> Dict[_T2, float]:
  ngrams_counter = Counter(x for y in ngrams.values() for x in y)
  vals = list(ngrams_counter.values())
  sum_vals = sum(vals)
  distr: Dict[_T2, float] = {k: v / sum_vals for k, v in ngrams_counter.items()}
  return distr


def get_reverse_distribution(ngrams: Dict[_T1, List[_T2]]) -> Dict[_T2, float]:
  ngrams_counter = Counter(x for y in ngrams.values() for x in y)
  keys_sorted = list(sorted(ngrams_counter.keys()))
  od = OrderedDict({k: ngrams_counter[k] for k in keys_sorted})
  vals = list(od.values())
  sum_vals = sum(vals)
  distr: Dict[_T2, float] = {k: vals[i] / sum_vals for i, k in enumerate(reversed(keys_sorted))}
  return distr


def get_filtered_ngrams(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]]) -> OrderedDictType[int, List[Tuple]]:
  assert isinstance(data, OrderedDict)

  logger = getLogger(__name__)
  logger.info(f"Calculating {n_gram}-grams...")
  available_ngrams: OrderedDictType[int, List[Tuple]] = OrderedDict({
    k: get_ngrams(v, n_gram) for k, v in data.items()
  })

  occurring_symbols = {x for y in data.values() for x in y}
  occurring_symbols_count = len(occurring_symbols)
  occurring_ngrams = {x for y in available_ngrams.values() for x in y}
  occurring_ngrams_count = len(occurring_ngrams)

  logger.info(
      f"Theoretically, the maximum amount of unique {n_gram}-grams is: {occurring_symbols_count ** n_gram}.")

  if occurring_symbols_count > 0:
    logger.info(
      f"The amount of unique occurring {n_gram}-grams is: {occurring_ngrams_count} ({occurring_ngrams_count/(occurring_symbols_count ** n_gram)*100:.2f}%).")

  if ignore_symbols is not None:
    occuring_ignore_symbols = occurring_symbols.intersection(ignore_symbols)

    if len(occuring_ignore_symbols) > 0:
      logger.info(
        f"Removing {n_gram}-grams which contain: {' '.join(list(sorted(occuring_ignore_symbols)))}...")
      available_ngrams: OrderedDictType[int, List[Tuple]] = OrderedDict({
        k: filter_ngrams(v, occuring_ignore_symbols) for k, v in available_ngrams.items()
      })

      new_occurring_ngrams = {x for y in available_ngrams.values() for x in y}
      new_occurring_ngrams_count = len(new_occurring_ngrams)

      logger.info(
          f"Removed {occurring_ngrams_count - new_occurring_ngrams_count} unique {n_gram}-gram(s).")

  return available_ngrams


def get_first_percent(data: OrderedSet, percent: float) -> OrderedSet:
  proportion = len(data) / 100 * percent
  # rounding is strange see: https://docs.python.org/3/library/functions.html#round
  percent_count = round(proportion)
  res = data[:percent_count]
  return res


def get_n_divergent_seconds(durations_s: OrderedDictType[_T1, float], seconds: float, n: int) -> List[OrderedSet[_T1]]:
  total_dur = sum(durations_s.values())
  assert seconds <= total_dur
  dur_to_fill = n * seconds
  stack_times = math.ceil(dur_to_fill / total_dur)
  data_keys = list(durations_s.keys())
  data_keys *= stack_times
  step_length = round(total_dur / n)

  selected, _ = get_until_sum_set(
      data_keys, until_values=durations_s, until_value=seconds)
  res: List[OrderedSet[_T1]] = [selected]

  for _ in range(n - 1):
    start_index = get_next_start_index(step_length, durations_s, list(res[-1]), data_keys)
    selected, _ = get_until_sum_set(
      data_keys[start_index:], until_values=durations_s, until_value=seconds)
    res.append(selected)
  return res


def get_next_start_index(step_length: int, durations_s: OrderedDictType[_T1, float], prev_vec: List[_T1], data_keys: List[_T1]) -> int:
  # der Startindex soll auf das Element in data_keys referieren, das zu dem ersten Element in prev_vec mindestens den Abstand step_length hat (d.h. genau diesen Abstand hat oder das erste Element ist, für das dieser Abstand überschritten wird). Abstand ist hierbei definiert als die aufsummierten Durations vom ersten Element in prev_vec (dieses wird nicht mit einberechnet) bis zum Element, für das der Abstand berechnet wird (dieses wird mit einberechnet).
  # Falls kein Element in prev_vec einen Abstand >= step_length vom 1. Element in prev_vec aus gesehen hat, so soll der Index des nächsten Eintrags in data_keys zurückgegeben werden
  """
  Bsp.: step_length = 4
       durations = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 2, 6: 2}
       prev_vec = [0, 1, 2, 3]
       Es ist das Element in {0,...,6} gesucht, das zu 0 mindestens Abstand 4 hat
       Das wäre hier die 2, denn Entfernung(0,2) = dur(1) + dur(2) = 2+3>4

  Bsp. 2: step_length = 4
          durations = {index: 1 for index in range(8)}
          prev_vec = [0, 1, 2, 3]
          data_keys = [0, 1, 2, 3, 4, 5, 6, 7]
          Hier ist dur(1) + dur(2) + dur(3) < 4, daher wird der Index des auf 3 in data_keys folgenden Elements zurückgegeben (also die 4)
  """
  assert len(prev_vec) > 0
  assert len(data_keys) > 0
  dur_sum = 0
  index = 0
  while dur_sum < step_length:
    index += 1
    if index == len(prev_vec):
      prev_element = prev_vec[-1]
      start_index = data_keys.index(prev_element) + 1
      return start_index
    dur_sum += durations_s[prev_vec[index]]
  start_element = prev_vec[index]
  start_index = data_keys.index(start_element)
  return start_index


# def ignore_ngrams(available_ngrams: OrderedDictType[int, List[Tuple]], ignore_symbols: Set[str]):
#   logger = getLogger(__name__)
#   if len(ignore_symbols) > 0:
#     occurring_ngrams = {x for y in available_ngrams.values() for x in y}
#     occurring_ngrams_count = len(occurring_ngrams)
#     logger.info(
#       f"Removing entries which contain any of these symbols: {' '.join(list(sorted(ignore_symbols)))}...")
#     available_ngrams: OrderedDictType[int, List[Tuple]] = OrderedDict({
#       k: filter_ngrams(v, ignore_symbols) for k, v in available_ngrams.items()
#     })

#     new_occurring_ngrams = {x for y in available_ngrams.values() for x in y}
#     new_occurring_ngrams_count = len(new_occurring_ngrams)
#     logger.info(
#         f"Removed {occurring_ngrams_count - new_occurring_ngrams_count} unique {n_gram}-gram(s).")
