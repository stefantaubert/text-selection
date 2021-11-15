import itertools
import math
from collections import Counter, OrderedDict
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from typing import Dict, Iterator, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ordered_set import OrderedSet
from scipy.stats import entropy
from tqdm import tqdm

from text_selection.greedy_kld_applied import get_chunksize_for_data
from text_selection.greedy_kld_methods import (dict_to_array_ordered,
                                               sync_dict_keys_to_keys_inplace)
from text_selection.selection import (FirstKeySelector, KeySelector,
                                      SelectionMode, order_keys, select_key)
from text_selection.utils import DurationBoundary, get_ngrams

NGramNr = int
NGramNrs = np.ndarray
NGram = Tuple[str, ...]


def get_distribution_array(counts: np.ndarray) -> np.ndarray:
  # assert all(x >= 0 for x in counts) # too slow
  sum_counts = np.sum(counts)
  new_dist = np.divide(counts, sum_counts)
  return new_dist


class KldIterator(Iterator[int]):
  def __init__(self, data: np.ndarray, target_dist: np.ndarray, preselection: np.ndarray, key_selector: KeySelector, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
    super().__init__()
    self.key_selector = key_selector
    # defines the order for what the selection is based on
    self.available_data_keys_ordered = OrderedSet(range(len(data)))
    self.data = data
    self.covered_array = preselection.copy()
    self.target_dist = target_dist
    self.n_jobs = n_jobs
    self.maxtasksperchild = maxtasksperchild
    self.chunksize = chunksize
    self.batches = batches

  def __iter__(self) -> Iterator[int]:
    return self

  def get_current_kld(self) -> float:
    current_distribution = get_distribution_array(self.covered_array)
    kld = entropy(current_distribution, self.target_dist)
    return kld

  def __next__(self) -> int:
    if len(self.available_data_keys_ordered) == 0:
      raise StopIteration()

    divergences = get_divergences_np_based2(
      data=self.data,
      keys=self.available_data_keys_ordered,
      covered_counts=self.covered_array,
      target_dist=self.target_dist,
      maxtasksperchild=self.maxtasksperchild,
      n_jobs=self.n_jobs,
      chunksize=self.chunksize,
      batches=self.batches,
    )

    # minima_indicies = np.argmin(divergences, axis=0)
    min_div = divergences.min()
    minima_indicies = np.flatnonzero(divergences == min_div)
    potential_keys = OrderedSet(
      self.available_data_keys_ordered[index] for index in minima_indicies
    )

    # potential_keys = get_smallest_divergence_keys(divergences)
    # potential_keys = order_keys(potential_keys, self.available_data_keys_ordered)

    if len(potential_keys) > 1:
      logger = getLogger(__name__)
      logger.info(f"Found {len(potential_keys)} candidates for the current iteration.")

    selected_key = self.key_selector.select_key(potential_keys)
    assert 0 <= selected_key < len(self.data)
    self.covered_array += self.data[selected_key]
    self.available_data_keys_ordered.remove(selected_key)
    return selected_key


def get_uniform_distribution_ngrams(ngrams: OrderedSet[NGramNr]) -> Dict[NGramNr, float]:
  if len(ngrams) == 0:
    return dict()
  distr = 1 / len(ngrams)
  res: Dict[NGramNr, float] = {ngram_nr: distr for ngram_nr in ngrams}
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


def greedy_kld_uniform_ngrams_seconds_with_preselection_perf(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], select_from_durations_s: Dict[int, float], seconds: float, duration_boundary: DurationBoundary, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> OrderedSet[int]:
  logger = getLogger(__name__)

  select_from_keys = get_duration_keys(select_from_durations_s, select_from_keys, duration_boundary)

  logger.info(f"Collecting data symbols...")
  data_symbols = get_unique_symbols(data, select_from_keys)
  logger.info(f"Collecting preselection symbols...")
  preselection_symbols = get_unique_symbols(data, preselection_keys)
  target_symbols = data_symbols | preselection_symbols
  if ignore_symbols is not None:
    target_symbols -= ignore_symbols

  logger.info(f"Calculating all possible {n_gram}-grams...")
  possible_ngrams = list(tqdm(
    itertools.product(sorted(target_symbols), repeat=n_gram),
    total=len(target_symbols) ** n_gram,
  ))
  ngram_nr_to_ngram: Dict[NGram, NGramNr] = {
    k: i for i, k in enumerate(possible_ngrams)
  }
  all_ngram_nrs = OrderedSet(ngram_nr_to_ngram.values())

  logger.info(f"Calculating {n_gram}-grams...")
  data_ngrams = get_ngrams_from_data(
    data=data,
    keys=select_from_keys,
    n_gram=n_gram,
    ngram_nr_to_ngram=ngram_nr_to_ngram,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  np_counts = get_counts_array(
    data_ngrams,
    data_ngrams.keys(),
    ngram_nrs=all_ngram_nrs,
    n_jobs=n_jobs,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
    batches=batches,
  )
  del data_ngrams

  # remove empty rows
  empty_entry_ids = np.where(~np_counts.any(axis=1))[0]
  if len(empty_entry_ids > 0):
    logger.info(f"Removing {len(empty_entry_ids)} empty row(s) out of {len(np_counts)} rows...")
    for empty_entry_id in reversed(sorted(empty_entry_ids)):
      remove_key = select_from_keys[empty_entry_id]
      select_from_keys.remove(remove_key)
    np_counts = np.delete(np_counts, empty_entry_ids, axis=0)
    logger.info("Done.")

  # preselected_ngrams = tuple(ngram for ngrams in preselection_ngrams.values() for ngram in ngrams)
  # preselection_counts = get_count_array(preselected_ngrams, all_ngram_nrs)

  preselection_ngrams = get_ngrams_from_data(
    data=data,
    keys=preselection_keys,
    n_gram=n_gram,
    ngram_nr_to_ngram=ngram_nr_to_ngram,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  pre_np_counts = get_counts_array(
    preselection_ngrams,
    preselection_ngrams.keys(),
    ngram_nrs=all_ngram_nrs,
    n_jobs=n_jobs,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
    batches=batches,
  )
  del preselection_ngrams

  preselection_counts: NDArray = np.sum(pre_np_counts, axis=0)
  del pre_np_counts

  # remove empty columns, can only occur on n_gram > 1
  data_counts: NDArray = np.sum(np_counts, axis=0)
  all_counts: NDArray = data_counts + preselection_counts
  remove_ngrams = np.where(all_counts == 0)[0]
  if len(remove_ngrams) > 0:
    logger.info(f"Removing {len(remove_ngrams)} out of {len(all_ngram_nrs)} columns...")
    for remove_ngram_nr in remove_ngrams:
      all_ngram_nrs.remove(remove_ngram_nr)
    np_counts = np.delete(np_counts, remove_ngrams, axis=1)
    preselection_counts = np.delete(preselection_counts, remove_ngrams, axis=0)
    logger.info("Done.")

  ngrams_str = [f"\"{''.join(ngram)}\"" for ngram,
                ngram_nr in ngram_nr_to_ngram.items() if ngram_nr in all_ngram_nrs]
  logger.info(
    f"Obtained {len(all_ngram_nrs)} {n_gram}-gram(s): {', '.join(ngrams_str)}.")

  uniform_distr = get_uniform_distribution_ngrams(all_ngram_nrs)
  if len(uniform_distr) > 0:
    logger.info(f"Target (uniform) distribution: {list(uniform_distr.values())[0]}")
  target_distribution_array = dict_to_array_ordered(uniform_distr, all_ngram_nrs)

  preselected_ngrams_count: int = np.sum(preselection_counts, axis=0)
  if preselected_ngrams_count > 0:
    preselection_distr = get_distribution_array(preselection_counts)
    preselection_kld = entropy(preselection_distr, target_distribution_array)
    logger.info(f"Preselection Kullback-Leibler divergence: {preselection_kld}")

  until_values = np.zeros(shape=(len(np_counts)))
  for index, k in enumerate(select_from_keys):
    assert k in select_from_durations_s
    until_values[index] = select_from_durations_s[k]

  iterator = KldIterator(
    data=np_counts,
    target_dist=target_distribution_array,
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
    n_jobs=n_jobs,
    batches=batches,
    preselection=preselection_counts,
    key_selector=FirstKeySelector(),
  )

  greedy_selected, enough_data_was_available = iterate_durations(
    iterator=iterator,
    until_value=seconds,
    until_values=until_values,
  )

  if not enough_data_was_available:
    logger.warning(
      f"Aborted since no further data had been available!")

  # greedy_selected = sort_greedy_kld_until_with_preselection_np_based(
  #   data=np_counts,
  #   target_dist=target_distribution_array,
  #   chunksize=chunksize,
  #   maxtasksperchild=maxtasksperchild,
  #   n_jobs=n_jobs,
  #   batches=batches,
  #   preselection=preselection_counts,
  # )

  logger.info(f"Obtained Kullback-Leibler divergence: {iterator.get_current_kld()}")

  result = OrderedSet([select_from_keys[index] for index in greedy_selected])

  return result

# class DurationIterator(Iterator[int]):
#   def __init__(self, iterator: Iterator[int], until_values: np.ndarray, until_value: Union[int, float]) -> None:
#     super().__init__()
#     self.iterator = iter(iterator)
#     self.max_until = sum(until_values)
#     self.adjusted_until = round(min(until_value, self.max_until))
#     current_total = 0.0

#   def __iter__(self) -> Iterator[int]:
#     return self
#   def __next__(self) -> int:
#     selected_key = next(self.iterator)
#     assert 0 <= selected_key < len(self.until_values)
#     selected_until_value = self.until_values[selected_key]
#     new_total = self.current_total + selected_until_value
#     if new_total <= self.until_value:
#       return selected_key,
#       iterated_values.append(selected_key)
#       current_total = new_total
#       progress_bar.update(round(selected_until_value))
#       if current_total == until_value:
#         enough_data_was_available = True
#     else:
#       enough_data_was_available = True
#       break
#     return value


def iterate_durations(iterator: Iterator[int], until_values: np.ndarray, until_value: Union[int, float]) -> Tuple[List[int], bool]:
  iterated_values: List[int] = []
  enough_data_was_available = False
  max_until = sum(until_values)
  adjusted_until = round(min(until_value, max_until))
  current_total = 0.0
  with tqdm(total=adjusted_until, initial=round(current_total)) as progress_bar:
    for selected_key in iterator:
      assert 0 <= selected_key < len(until_values)
      selected_until_value = until_values[selected_key]
      new_total = current_total + selected_until_value
      if new_total <= until_value:
        iterated_values.append(selected_key)
        current_total = new_total
        progress_bar.update(round(selected_until_value))
        if current_total == until_value:
          enough_data_was_available = True
      else:
        enough_data_was_available = True
        break
  # Selected: {current_total:.1f}/{until_value:.1f} ({current_total/until_value*100:.2f}%).
  return iterated_values, enough_data_was_available


process_data_ngrams: Dict[int, NGramNr] = None
process_ngram_nrs: OrderedSet[NGramNr] = None


def init_pool_np_counts(data_ngrams: Dict[int, NGramNr], ngram_nrs: OrderedSet[NGramNr]) -> None:
  global process_data_ngrams
  global process_ngram_nrs
  process_data_ngrams = data_ngrams
  process_ngram_nrs = ngram_nrs


def main_np_counts(i_k: Tuple[int, int]) -> Tuple[int, np.ndarray]:
  global process_data_ngrams
  global process_ngram_nrs
  i, k = i_k
  ngrams = process_data_ngrams[k]
  counts = get_count_array(ngrams, process_ngram_nrs)
  return i, counts


def log_mp_params(n_jobs: int, chunksize: int, maxtasksperchild: Optional[int], data_count: int) -> None:
  logger = getLogger(__name__)
  logger.info(
    f"Using {n_jobs} processes with chunks of size {chunksize} for {data_count} utterances and maxtask per child: {maxtasksperchild}.")


def get_counts_array(data_ngrams: Dict[int, NGramNr], select_from_keys: Set[int], ngram_nrs: OrderedSet[NGramNr], n_jobs: int, chunksize: Optional[int], maxtasksperchild: Optional[int], batches: Optional[int]) -> NDArray:
  logger = getLogger(__name__)
  logger.info("Calculating counts array...")
  np_counts = np.zeros(shape=(len(select_from_keys), len(ngram_nrs)), dtype=np.uint32)

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
  all_keys_exist_in_durations = len(durations.keys() - keys) == 0
  assert all_keys_exist_in_durations
  boundary_min, boundary_max = boundary
  logger.info("Getting entries maching duration boundary...")
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


process_ngram_nr_to_ngram: Dict[Tuple[str, ...], NGramNr] = None
process_data: OrderedDictType[int, Tuple[str, ...]] = None


def main(key: int, n: int) -> Tuple[int, NGramNrs]:
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


def init_pool_ngrams(data: OrderedDictType[int, Tuple[str, ...]], ngram_nr_to_ngram: Dict[Tuple[str, ...], NGramNr]) -> None:
  global process_data
  global process_ngram_nr_to_ngram
  process_data = data
  process_ngram_nr_to_ngram = ngram_nr_to_ngram


def get_unique_symbols(data: Dict[int, Tuple[str, ...]], keys: Set[int]) -> Set[str]:
  occurring_symbols = {symbol for key in tqdm(keys, total=len(keys)) for symbol in data[key]}
  return occurring_symbols


def get_ngrams_from_data(data: OrderedDictType[int, Tuple[str, ...]], keys: Set[int], ngram_nr_to_ngram: Dict[Tuple[str, ...], NGramNrs], n_gram: int, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> Dict[int, NGramNrs]:
  logger = getLogger(__name__)

  if len(data) == 0:
    return dict()

  method_proxy = partial(
    main,
    n=n_gram,
  )

  logger.info(f"Getting {n_gram}-grams...")
  chunksize = get_chunksize(len(keys), n_jobs, chunksize, batches)
  log_mp_params(n_jobs, chunksize, maxtasksperchild, len(keys))

  with Pool(
      processes=n_jobs,
      initializer=init_pool_ngrams,
      initargs=(data, ngram_nr_to_ngram),
      maxtasksperchild=maxtasksperchild,
    ) as pool:
    available_ngrams: Dict[int, NGramNrs] = dict(tqdm(
      pool.imap_unordered(method_proxy, keys, chunksize=chunksize),
      total=len(keys),
    ))

  return available_ngrams


def get_count_array(ngrams: NGramNrs, target_symbols_ordered: OrderedSet[NGramNr]) -> np.ndarray:
  counts = Counter(ngrams)
  sync_dict_keys_to_keys_inplace(counts, target_symbols_ordered)
  return dict_to_array_ordered(counts, target_symbols_ordered)


def sort_greedy_kld_until_with_preselection_np_based(data: np.ndarray, target_dist: np.ndarray, until_values: np.ndarray, until_value: Union[float, int], preselection: np.ndarray, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> OrderedSet[int]:
  logger = getLogger(__name__)
  selection_mode = SelectionMode.FIRST
  result: OrderedSet[int] = OrderedSet()
  # defines the order for what the selection is based on
  available_data_keys_ordered = OrderedSet(range(len(data)))
  covered_array = preselection.copy()

  logger.info("Selecting data...")

  chunksize = get_chunksize(len(data), n_jobs, chunksize, batches)
  log_mp_params(n_jobs, chunksize, maxtasksperchild, len(data))

  max_until = sum(until_values)
  adjusted_until = round(min(until_value, max_until))
  current_total = 0.0
  with tqdm(total=adjusted_until, initial=round(current_total)) as progress_bar:
    while True:
      if len(available_data_keys_ordered) == 0:
        logger.warning(
          f"Aborting selection as no further data is available! Selected: {current_total:.1f}/{until_value:.1f} ({current_total/until_value*100:.2f}%).")
        break
      potential_keys = get_utterance_with_min_kld_np_based(
        data=data,
        keys=available_data_keys_ordered,
        covered_counts=covered_array,
        target_dist=target_dist,
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
        selected_count_array = data[selected_key]
        covered_array += selected_count_array
        current_total = new_total
        available_data_keys_ordered.remove(selected_key)
        progress_bar.update(round(selected_until_value))
        if current_total == until_value:
          break
      else:
        break

  final_distr = get_distribution_array(covered_array)
  final_kld = entropy(final_distr, target_dist)
  logger.info(f"Obtained Kullback-Leibler divergence: {final_kld}")

  return result


def get_divergences_np_based(data: np.ndarray, keys: OrderedSet[int], covered_counts: np.ndarray, target_dist: np.ndarray, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> Dict[int, float]:
  # logger.debug(f"Using {thread_count} threads with {chunksize} chunks...")
  # logger.info("Calculating Kullback-Leibler divergences...")

  chunksize = get_chunksize(len(keys), n_jobs, chunksize, batches)
  log_mp_params(n_jobs, chunksize, maxtasksperchild, len(keys))

  with Pool(
    processes=n_jobs,
    initializer=init_pool_np_based,
    initargs=(data, covered_counts, target_dist),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    result: Dict[int, float] = dict(pool.imap_unordered(
      get_divergence_for_utterance_np_based, keys, chunksize=chunksize
    ))

  return result


def get_divergences_np_based2(data: np.ndarray, keys: OrderedSet[int], covered_counts: np.ndarray, target_dist: np.ndarray, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> np.ndarray:
  # logger.debug(f"Using {thread_count} threads with {chunksize} chunks...")
  # logger.info("Calculating Kullback-Leibler divergences...")

  chunksize = get_chunksize(len(keys), n_jobs, chunksize, batches)
  log_mp_params(n_jobs, chunksize, maxtasksperchild, len(keys))

  result = np.zeros(shape=(len(keys)))

  with Pool(
    processes=n_jobs,
    initializer=init_pool_np_based,
    initargs=(data, covered_counts, target_dist),
    maxtasksperchild=maxtasksperchild,
  ) as pool:
    iterator = pool.imap_unordered(
      get_divergence_for_utterance_np_based2, enumerate(keys), chunksize=chunksize
    )
    for index, kld in iterator:
      result[index] = kld

  return result


process_data_np_based: np.ndarray = None
process_covered_counts_np_based: np.ndarray = None
process_target_dist_np_based: np.ndarray = None


def init_pool_np_based(data: np.ndarray, covered_counts: np.ndarray, target_dist: np.ndarray) -> None:
  global process_data_np_based
  global process_covered_counts_np_based
  global process_target_dist_np_based
  process_data_np_based = data
  process_covered_counts_np_based = covered_counts
  process_target_dist_np_based = target_dist


def get_divergence_for_utterance_np_based2(index_key: Tuple[int, int]) -> Tuple[int, float]:
  global process_data_np_based
  global process_covered_counts_np_based
  global process_target_dist_np_based
  index, key = index_key
  assert key < len(process_data_np_based)
  utterance_counts = process_data_np_based[key]
  counts = process_covered_counts_np_based + utterance_counts
  distr = get_distribution_array(counts)
  kld = get_kld(distr, process_target_dist_np_based)
  return index, kld


def get_divergence_for_utterance_np_based(key: int) -> Tuple[int, float]:
  global process_data_np_based
  global process_covered_counts_np_based
  global process_target_dist_np_based
  assert key < len(process_data_np_based)
  utterance_counts = process_data_np_based[key]
  counts = process_covered_counts_np_based + utterance_counts
  distr = get_distribution_array(counts)
  kld = get_kld(distr, process_target_dist_np_based)
  return key, kld


def get_smallest_divergence_keys(divergences: Dict[int, float]) -> Set[int]:
  assert len(divergences) > 0

  minimum_divergence = min(divergences.values())
  all_with_minimum_divergence = {
    key for key, divergence in divergences.items()
    if divergence == minimum_divergence
  }
  return all_with_minimum_divergence


def get_kld(dist: np.ndarray, target_dist: np.ndarray) -> float:
  none_of_targed_ngrams_exist = all(np.isnan(dist))
  if none_of_targed_ngrams_exist:
    return math.inf

  res = entropy(dist, target_dist)
  return res
