import itertools
import math
from collections import Counter, OrderedDict
from functools import partial
from logging import getLogger
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple, Union

import numpy as np
from ordered_set import OrderedSet
from scipy.stats import entropy
from tqdm import tqdm

from text_selection.greedy_kld_applied import get_chunksize_for_data
from text_selection.greedy_kld_methods import (__get_distribution,
                                               dict_to_array_ordered,
                                               get_uniform_distribution,
                                               sync_dict_keys_to_keys_inplace)
from text_selection.selection import SelectionMode, order_keys, select_key
from text_selection.utils import (DurationBoundary,
                                  filter_data_durations_number_inplace,
                                  get_distribution, get_ngrams)

NGram = int
NGrams = Tuple[NGram, ...]


def greedy_kld_uniform_ngrams_seconds_with_preselection_perf(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], durations_s: Dict[int, float], seconds: float, preselection: OrderedDictType[int, List[str]], duration_boundary: DurationBoundary, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int]) -> OrderedSet[int]:
  data_ngrams = get_filtered_ngrams_numbers(data, n_gram, ignore_symbols)
  filter_data_durations_number_inplace(data_ngrams, durations_s, duration_boundary)
  preselection_ngrams = get_filtered_ngrams_numbers(preselection, n_gram, ignore_symbols)

  logger = getLogger(__name__)
  uniform_distr = get_uniform_distribution(data_ngrams)
  if len(uniform_distr) > 0:
    logger.info(f"Target uniform distribution: {list(uniform_distr.values())[0]}")
  if chunksize is None:
    chunksize = get_chunksize_for_data(data_ngrams, n_jobs)

  greedy_selected = sort_greedy_kld_until_with_preselection(
    data=data_ngrams,
    target_dist=uniform_distr,
    until_values=durations_s,
    until_value=seconds,
    preselection=preselection_ngrams,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
  )

  return greedy_selected


process_ngram_nr_to_ngram: Dict[Tuple[str, ...], NGram] = None
process_data: OrderedDictType[int, Tuple[str, ...]] = None


def main(key: int, n: int) -> Tuple[int, NGrams]:
  global process_data
  global process_ngram_nr_to_ngram
  n_gram = process_data[key]

  ngrams = get_ngrams(n_gram, n)
  ngrams_int = tuple(process_ngram_nr_to_ngram[ngram] for ngram in ngrams)
  return key, ngrams_int


def init_pool_ngrams(data: OrderedDictType[int, Tuple[str, ...]], ngram_nr_to_ngram: Dict[Tuple[str, ...], NGram]) -> None:
  global process_data
  global process_ngram_nr_to_ngram
  process_data = data
  process_ngram_nr_to_ngram = ngram_nr_to_ngram


def get_filtered_ngrams_numbers(data: OrderedDictType[int, Tuple[str, ...]], n_gram: int, ignore_symbols: Optional[Set[str]]) -> OrderedDictType[int, NGrams]:
  assert isinstance(data, OrderedDict)

  logger = getLogger(__name__)
  logger.info(f"Calculating {n_gram}-grams...")

  #ngram_nr_to_ngram: Dict[Tuple[str, ...], NGram] = {}

  logger.info(f"Collecting all symbols...")
  occurring_symbols = {x for y in tqdm(data.values(), total=len(data)) for x in y}

  logger.info(f"Calculating all possible {n_gram}-grams...")
  possible_ngrams = list(tqdm(itertools.permutations(
    occurring_symbols, r=n_gram), total=len(occurring_symbols) ** n_gram))

  ngram_nr_to_ngram: OrderedDictType[int, NGrams] = OrderedDict([
    (k, i) for i, k in enumerate(possible_ngrams)
  ])

  n_jobs = cpu_count()
  chunksize = get_chunksize_for_data(data, n_jobs)
  method_proxy = partial(
    main,
    n=n_gram,
  )
  logger.info(f"Getting {n_gram}-grams...")
  with Pool(
      processes=n_jobs,
      initializer=init_pool_ngrams,
      initargs=(data, ngram_nr_to_ngram),
      maxtasksperchild=None,
    ) as pool:
    available_ngrams: Dict[int, NGrams] = OrderedDict(tqdm(
      pool.imap_unordered(method_proxy, data.keys(), chunksize=chunksize),
      total=len(data),
    ))
  logger.info("Done.")

  # for k, v in tqdm(data.items(), total=len(data)):
  #   ngrams = get_ngrams(v, n_gram)
  #   ngrams_int = tuple(possible_ngrams[ngram] for ngram in ngrams)
  #   # for ngram in ngrams:
  #   #   if ngram in ngram_nr_to_ngram:
  #   #     ngrams_int.append(ngram_nr_to_ngram[ngram])
  #   #   else:
  #   #     new_nr = len(ngram_nr_to_ngram)
  #   #     ngram_nr_to_ngram[ngram] = new_nr
  #   #     ngrams_int.append(new_nr)
  #   available_ngrams[k] = ngrams_int

  occurring_symbols_count = len(occurring_symbols)
  occurring_ngrams_count = len(ngram_nr_to_ngram)

  logger.info(
      f"Theoretically, the maximum amount of unique {n_gram}-grams is: {occurring_symbols_count ** n_gram}.")

  if occurring_symbols_count > 0:
    logger.info(
      f"The amount of unique occurring {n_gram}-grams is: {occurring_ngrams_count} ({occurring_ngrams_count/(occurring_symbols_count ** n_gram)*100:.2f}%).")

  if ignore_symbols is not None:
    occuring_ignore_symbols = occurring_symbols.intersection(ignore_symbols)

    if len(occuring_ignore_symbols) > 0:
      logger.info(
        f"Removing {n_gram}-grams which contain: \"{', '.join(list(sorted(occuring_ignore_symbols)))}\"...")

      ignore_ngram_nrs = {
        ngram_nr for ngram, ngram_nr in ngram_nr_to_ngram.items()
        if len(set(ngram).intersection(ignore_symbols)) > 0
      }

      for k, ngram_nrs in available_ngrams.items():
        ngram_nrs_filtered = tuple(
          ngram_nr for ngram_nr in ngram_nrs
          if ngram_nr not in ignore_ngram_nrs
        )
        changed_something = len(ngram_nrs_filtered) != len(ngram_nrs)
        if changed_something:
          available_ngrams[k] = ngram_nrs_filtered

      new_occurring_ngrams_count = len(ngram_nr_to_ngram) - len(ignore_ngram_nrs)

      logger.info(
          f"Removed {occurring_ngrams_count - new_occurring_ngrams_count} unique {n_gram}-gram(s).")

  return available_ngrams


def get_available_array(ngrams: NGrams, target_symbols_ordered: OrderedSet[NGram]) -> np.ndarray:
  counts = Counter(ngrams)
  sync_dict_keys_to_keys_inplace(counts, target_symbols_ordered)
  return dict_to_array_ordered(counts, target_symbols_ordered)


def sort_greedy_kld_until_with_preselection(data: OrderedDictType[int, NGrams], target_dist: Dict[NGram, float], until_values: Dict[int, Union[float, int]], until_value: Union[float, int], preselection: OrderedDictType[int, NGrams], n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> OrderedSet[int]:
  assert isinstance(data, OrderedDict)
  assert isinstance(preselection, OrderedDict)
  # The probability is really high that only one key is figured out, therefore it is useless to use any selection modes. If shortest or longest should be used the unfiltered count of symbols needs to be passed as extra parameter which increases complexity of the method.
  selection_mode = SelectionMode.FIRST
  logger = getLogger(__name__)
  result: OrderedSet[int] = OrderedSet()
  target_symbols_ordered: OrderedSet[NGram] = OrderedSet(sorted(target_dist.keys()))
  # defines the order for what the selection is based on
  available_data_keys_ordered = OrderedSet(data.keys())
  # all_occuring_values: Set[_T2] = {x for y in data.values() for x in y}
  # assert all_keys == all_occuring_values

  logger.info("Preparing data...")
  target_distribution_array = dict_to_array_ordered(target_dist, target_symbols_ordered)

  if len(preselection) == 0:
    covered_counter = {x: 0 for x in target_symbols_ordered}
    covered_array = dict_to_array_ordered(covered_counter, target_symbols_ordered)
  else:
    logger.info("Using preselected data.")
    preselected_ngrams = tuple(ngram for ngrams in preselection.values() for ngram in ngrams)
    covered_array = get_available_array(preselected_ngrams, target_symbols_ordered)
    preselection_distr = __get_distribution(covered_array)
    preselection_kld = entropy(preselection_distr, target_distribution_array)
    logger.info(f"Preselection Kullback-Leibler divergence: {preselection_kld}")

  logger.info("Selecting data...")
  max_until = sum(until_values.values())
  adjusted_until = round(min(until_value, max_until))
  current_total = 0.0
  with tqdm(total=adjusted_until, initial=round(current_total)) as progress_bar:
    while True:
      if len(available_data_keys_ordered) == 0:
        logger.warning(
          f"Aborting selection as no further data is available! Selected: {current_total:.1f}/{until_value:.1f} ({current_total/until_value*100:.2f}%).")
        break
      potential_keys = get_utterance_with_min_kld(
        data=data,
        keys=available_data_keys_ordered,
        target_symbols_ordered=target_symbols_ordered,
        covered_counts=covered_array,
        target_dist=target_distribution_array,
        maxtasksperchild=maxtasksperchild,
        n_jobs=n_jobs,
        chunksize=chunksize,
      )
      if len(potential_keys) > 1:
        logger.info(f"Found {len(potential_keys)} candidates for the current iteration.")
      potential_keys_ordered = order_keys(potential_keys, available_data_keys_ordered)
      selected_key = select_key(potential_keys_ordered, unit_counts=None, mode=selection_mode)
      selected_until_value = until_values[selected_key]
      new_total = current_total + selected_until_value
      if new_total <= until_value:
        result.add(selected_key)
        selected_count_array = get_available_array(data[selected_key], target_symbols_ordered)
        covered_array += selected_count_array
        current_total = new_total
        available_data_keys_ordered.remove(selected_key)
        progress_bar.update(round(selected_until_value))
        if current_total == until_value:
          break
      else:
        break

  final_distr = __get_distribution(covered_array)
  final_kld = entropy(final_distr, target_distribution_array)
  logger.info(f"Obtained Kullback-Leibler divergence: {final_kld}")

  return result


def get_utterance_with_min_kld(data: Dict[int, NGrams], keys: Set[int], target_symbols_ordered: OrderedSet[NGram], covered_counts: np.ndarray, target_dist: np.ndarray, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Set[int]:
  divergences = get_divergences(data, keys, target_symbols_ordered, covered_counts, target_dist,
                                n_jobs, maxtasksperchild, chunksize)
  all_with_minimum_divergence = get_smallest_divergence_keys(divergences)
  return all_with_minimum_divergence


def get_smallest_divergence_keys(divergences: Dict[int, float]) -> Set[int]:
  assert len(divergences) > 0
  minimum_divergence = min(divergences.values())
  all_with_minimum_divergence = {
    key for key, divergence in divergences.items()
    if divergence == minimum_divergence
  }
  return all_with_minimum_divergence


def get_divergences(data: Dict[int, NGrams], keys: Set[int], target_symbols_ordered: OrderedSet[NGram], covered_counts: np.ndarray, target_dist: np.ndarray, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> Dict[int, float]:
  # logger.debug(f"Using {thread_count} threads with {chunksize} chunks...")
  # logger.info("Calculating Kullback-Leibler divergences...")
  with Pool(
    processes=n_jobs,
    initializer=init_pool,
    initargs=(data, covered_counts, target_dist, target_symbols_ordered),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    result: Dict[int, float] = dict(pool.imap_unordered(
        get_divergence_for_utterance, keys, chunksize=chunksize
    ))

  return result


process_data: Dict[int, NGrams] = None
process_covered_counts: np.ndarray = None
process_target_dist: np.ndarray = None
process_target_symbols_ordered: OrderedSet[NGram] = None


def init_pool(data: Dict[int, NGrams], covered_counts: np.ndarray, target_dist: np.ndarray, target_symbols_ordered: OrderedSet[NGram]) -> None:
  global process_data
  global process_covered_counts
  global process_target_dist
  global process_target_symbols_ordered
  process_data = data
  process_covered_counts = covered_counts
  process_target_dist = target_dist
  process_target_symbols_ordered = target_symbols_ordered


def get_divergence_for_utterance(key: int) -> Tuple[int, float]:
  global process_data
  global process_covered_counts
  global process_target_dist
  global process_target_symbols_ordered
  utterance = process_data[key]
  utterance_counts = get_available_array(utterance, process_target_symbols_ordered)
  counts = process_covered_counts + utterance_counts
  distr = __get_distribution(counts)
  kld = get_kld(distr, process_target_dist)
  return key, kld


def get_kld(dist: np.ndarray, target_dist: np.ndarray) -> float:
  none_of_targed_ngrams_exist = all(np.isnan(dist))
  if none_of_targed_ngrams_exist:
    return math.inf

  res = entropy(dist, target_dist)
  return res
