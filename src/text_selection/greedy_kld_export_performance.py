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
import scipy as sp
from ordered_set import OrderedSet
from scipy.stats import entropy
from tqdm import tqdm

from text_selection.greedy_kld_applied import get_chunksize_for_data
from text_selection.greedy_kld_methods import (__get_distribution,
                                               dict_to_array_ordered,
                                               get_uniform_distribution,
                                               sync_dict_keys_to_keys_inplace)
from text_selection.selection import SelectionMode, order_keys, select_key
from text_selection.utils import (DurationBoundary, filter_after_duration,
                                  filter_data_durations_number_inplace,
                                  get_distribution, get_ngrams)

NGram = int
NGrams = np.ndarray


def get_uniform_distribution_ngrams(ngrams: Dict[Tuple[str, ...], NGrams]) -> Dict[NGram, float]:
  if len(ngrams) == 0:
    return dict()
  distr = 1 / len(ngrams)
  res: Dict[NGram, float] = {ngram_nr: distr for _, ngram_nr in ngrams.items()}
  return res


def get_chunksize(data_count: int, n_jobs: int, chunksize: Optional[int], batches: Optional[int]) -> int:
  if batches is None:
    assert chunksize is not None
    assert chunksize > 0
    return chunksize

  if data_count == 0:
    return 1
  chunksize = math.ceil(data_count / n_jobs / batches)
  return chunksize


def greedy_kld_uniform_ngrams_seconds_with_preselection_perf(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], select_from_durations_s: Dict[int, float], seconds: float, duration_boundary: DurationBoundary, n_jobs: int, maxtasksperchild: int, chunksize: Optional[int], batches: Optional[int]) -> OrderedSet[int]:
  logger = getLogger(__name__)

  select_from_keys = get_duration_keys(select_from_durations_s, select_from_keys, duration_boundary)

  logger.info(f"Collecting data symbols...")
  data_symbols = get_unique_symbols(data, select_from_keys)
  logger.info(f"Collecting preselection symbols...")
  preselection_symbols = get_unique_symbols(data, preselection_keys)
  target_symbols = data_symbols | preselection_symbols
  if ignore_symbols is None:
    target_symbols -= ignore_symbols

  logger.info(f"Calculating {n_gram}-grams...")
  data_ngrams, ngram_nr_to_ngram = get_ngrams_from_data(
    data=data,
    keys=select_from_keys,
    n_gram=n_gram,
    target_symbols=target_symbols,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  # keymap = {
  #   i: data_key
  #   for i, data_key in enumerate(duration_keys)
  # }

  # tmp_array = np.array(list(data_ngrams.values()))

  #filter_data_durations_number_inplace(data_ngrams, durations_s, duration_boundary)
  preselection_ngrams, ngram_nr_to_ngram_preselection = get_ngrams_from_data(
    data=data,
    keys=preselection_keys,
    n_gram=n_gram,
    target_symbols=target_symbols,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  assert ngram_nr_to_ngram == ngram_nr_to_ngram_preselection
  all_n_grams = ngram_nr_to_ngram

  # if ignore_symbols is not None and len(ignore_symbols) > 0:
  #   ignore_ngrams_inplace(data_ngrams, ignore_symbols, ngram_nr_to_ngram)
  #   ignore_ngrams_inplace(preselection_ngrams, ignore_symbols, preselection_ngram_nr_to_ngram)

  logger = getLogger(__name__)
  uniform_distr = get_uniform_distribution_ngrams(all_n_grams)
  if len(uniform_distr) > 0:
    logger.info(f"Target uniform distribution: {list(uniform_distr.values())[0]}")

  if chunksize is None:
    chunksize = get_chunksize_for_data(data_ngrams, n_jobs)

  np_counts = get_counts_array(
    data_ngrams,
    select_from_keys,
    all_n_grams,
    n_jobs=n_jobs,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
    batches=batches,
  )

  greedy_selected = sort_greedy_kld_until_with_preselection(
    data=data_ngrams,
    np_counts=np_counts,
    data_key_order=select_from_keys,
    target_dist=uniform_distr,
    until_values=select_from_durations_s,
    until_value=seconds,
    preselection=preselection_ngrams,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
  )

  return greedy_selected


process_data_ngrams: Dict[int, NGram] = None
process_ngram_nrs: OrderedSet[NGram] = None


def init_pool_np_counts(data_ngrams: Dict[int, NGram], ngram_nrs: OrderedSet[NGram]) -> None:
  global process_data_ngrams
  global process_ngram_nrs
  process_data_ngrams = data_ngrams
  process_ngram_nrs = ngram_nrs


def main_np_counts(i_k: Tuple[int, int]) -> Tuple[int, np.ndarray]:
  global process_data_ngrams
  global process_ngram_nrs
  i, k = i_k

  ngrams = process_data_ngrams[k]
  counts = get_available_array(ngrams, process_ngram_nrs)
  return i, counts


def log_mp_params(n_jobs: int, chunksize: int, maxtasksperchild: int, data_count: int) -> None:
  logger = getLogger(__name__)
  logger.info(
    f"Using {n_jobs} processes with chunks of size {chunksize} for {data_count} utterances and maxtask per child: {maxtasksperchild}.")


def get_counts_array(data_ngrams: Dict[int, NGram], select_from_keys: Set[int], all_n_grams: Dict[Tuple[str, ...], NGram], n_jobs: int, chunksize: Optional[int], maxtasksperchild: int, batches: Optional[int]):
  logger = getLogger(__name__)
  logger.info("Calculating counts array..")
  ngram_nrs = OrderedSet(all_n_grams.values())
  np_counts = np.zeros(shape=(len(select_from_keys), len(ngram_nrs)), dtype=np.uint32)

  # for i, k in enumerate(tqdm(select_from_keys)):
  #   ngrams = data_ngrams[k]
  #   counts = get_available_array(ngrams, ngram_nrs)
  #   np_counts[i] = counts

  chunksize = get_chunksize(len(select_from_keys), n_jobs, chunksize, batches)
  log_mp_params(n_jobs, chunksize, maxtasksperchild, len(select_from_keys))

  with Pool(
    processes=n_jobs,
    initializer=init_pool_np_counts,
    initargs=(data_ngrams, ngram_nrs),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    with tqdm(total=len(select_from_keys)) as pbar:
      iterator = pool.imap_unordered(
        main_np_counts, enumerate(select_from_keys), chunksize=chunksize)
      for index, counts in iterator:
        np_counts[index] = counts
        pbar.update()

  return np_counts


def get_duration_keys(durations: Dict[int, float], keys: Set[int], boundary: DurationBoundary) -> OrderedSet[int]:
  logger = getLogger(__name__)
  boundary_min, boundary_max = boundary
  logger.info("Getting entries maching duratino boundary...")
  filtered_utterance_ids = get_keys_in_duration_boundary(
    durations, keys, boundary_min, boundary_max)
  not_selected_utterances_out_of_boundary = len(keys) - len(filtered_utterance_ids)

  if not_selected_utterances_out_of_boundary > 0:
    logger.warning(
        f"Missed out utterances due to duration boundary [{boundary_min},{boundary_max}): {not_selected_utterances_out_of_boundary}/{len(keys)} ({not_selected_utterances_out_of_boundary/len(keys)*100:.2f}%) -> retrieved {len(filtered_utterance_ids)} entries.")
  else:
    logger.debug(
      f"Didn't missed out any utterances through boundary [{boundary_min},{boundary_max}) -> kept {len(filtered_utterance_ids)} entries.")

  return filtered_utterance_ids


def get_keys_in_duration_boundary(corpus: Dict[int, float], keys: OrderedSet[int], min_duration_incl: float, max_duration_excl: float) -> OrderedSet[int]:
  assert min_duration_incl >= 0
  assert max_duration_excl >= 0

  filtered_utterance_indicies: OrderedSet[int] = OrderedSet()

  for utterance_id in tqdm(keys):
    assert utterance_id in corpus
    utterance_duration = corpus[utterance_id]
    if min_duration_incl <= utterance_duration < max_duration_excl:
      filtered_utterance_indicies.add(utterance_id)

  return filtered_utterance_indicies


process_ngram_nr_to_ngram: Dict[Tuple[str, ...], NGram] = None
process_data: OrderedDictType[int, Tuple[str, ...]] = None


def main(key: int, n: int) -> Tuple[int, NGrams]:
  global process_data
  global process_ngram_nr_to_ngram
  n_gram = process_data[key]

  ngrams = get_ngrams(n_gram, n)
  ngram_nrs = tuple(
    process_ngram_nr_to_ngram[ngram]
    for ngram in ngrams
    if ngram in process_ngram_nr_to_ngram
  )

  ngrams_int = np.array(ngram_nrs, dtype=np.uint32)

  return key, ngrams_int


def init_pool_ngrams(data: OrderedDictType[int, Tuple[str, ...]], ngram_nr_to_ngram: Dict[Tuple[str, ...], NGram]) -> None:
  global process_data
  global process_ngram_nr_to_ngram
  process_data = data
  process_ngram_nr_to_ngram = ngram_nr_to_ngram


def get_unique_symbols(data: Dict[int, Tuple[str, ...]], keys: Set[int]) -> Set[str]:
  occurring_symbols = {symbol for key in tqdm(keys, total=len(keys)) for symbol in data[key]}
  return occurring_symbols


def get_ngrams_from_data(data: OrderedDictType[int, Tuple[str, ...]], keys: Set[int], target_symbols: Set[str], n_gram: int, n_jobs: int, maxtasksperchild: int, chunksize: Optional[int], batches: Optional[int]) -> Tuple[Dict[int, NGrams], Dict[Tuple[str, ...], NGrams]]:
  logger = getLogger(__name__)

  logger.info(f"Calculating all possible {n_gram}-grams...")
  possible_ngrams = list(tqdm(itertools.permutations(
    target_symbols, r=n_gram), total=len(target_symbols) ** n_gram))

  ngram_nr_to_ngram: Dict[Tuple[str, ...], NGrams] = {
    k: i for i, k in enumerate(possible_ngrams)
  }

  if len(keys) == 0 or (chunksize is not None and chunksize <= 0):
    chunksize = 1
  if batches is not None:
    chunksize = math.ceil(len(keys) / n_jobs / batches)

  if chunksize == 0:
    chunksize = 1

  method_proxy = partial(
    main,
    n=n_gram,
  )
  logger.info(f"Getting {n_gram}-grams...")
  with Pool(
      processes=n_jobs,
      initializer=init_pool_ngrams,
      initargs=(data, ngram_nr_to_ngram),
      maxtasksperchild=maxtasksperchild,
    ) as pool:
    available_ngrams: Dict[int, NGrams] = OrderedDict(tqdm(
      pool.imap_unordered(method_proxy, keys, chunksize=chunksize),
      total=len(keys),
    ))
  logger.info("Done.")

  return available_ngrams, ngram_nr_to_ngram


def get_available_array(ngrams: NGrams, target_symbols_ordered: OrderedSet[NGram]) -> np.ndarray:
  counts = Counter(ngrams)
  sync_dict_keys_to_keys_inplace(counts, target_symbols_ordered)
  return dict_to_array_ordered(counts, target_symbols_ordered)


def sort_greedy_kld_until_with_preselection(data: OrderedDictType[int, NGrams], np_counts: np.ndarray, data_key_order: OrderedSet[int], target_dist: Dict[NGram, float], until_values: Dict[int, Union[float, int]], until_value: Union[float, int], preselection: OrderedDictType[int, NGrams], n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> OrderedSet[int]:
  assert isinstance(data, OrderedDict)
  assert isinstance(preselection, OrderedDict)
  # The probability is really high that only one key is figured out, therefore it is useless to use any selection modes. If shortest or longest should be used the unfiltered count of symbols needs to be passed as extra parameter which increases complexity of the method.
  selection_mode = SelectionMode.FIRST
  logger = getLogger(__name__)
  result: OrderedSet[int] = OrderedSet()
  target_symbols_ordered: OrderedSet[NGram] = OrderedSet(sorted(target_dist.keys()))
  # defines the order for what the selection is based on
  available_data_keys_ordered = data_key_order.copy()
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
  assert key in process_data
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
