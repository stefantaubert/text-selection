import itertools
import math
import multiprocessing
from collections import Counter, OrderedDict
from functools import partial
from logging import getLogger
from multiprocessing import Pool, pool
from typing import Dict, Generator, Iterable, Iterator, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ordered_set import OrderedSet
from scipy.stats import entropy
from tqdm import tqdm

from text_selection.selection import FirstKeySelector, KeySelector
from text_selection.utils import (DurationBoundary, get_ngrams,
                                  get_ngrams_generator, log_mp_params)

NGramNr = int
NGramNrs = np.ndarray
NGram = Tuple[str, ...]


def get_distribution_array(counts: np.ndarray) -> np.ndarray:
  # assert all(x >= 0 for x in counts) # too slow
  sum_counts = np.sum(counts)
  new_dist = np.divide(counts, sum_counts)
  return new_dist


class KldCoreIterator(Iterator[int]):
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
    self.pool: pool.Pool = None

  def start(self):
    self.pool = Pool(
      processes=self.n_jobs,
      initializer=init_pool_np_based,
      initargs=(self.data, self.covered_array, self.target_dist),
      maxtasksperchild=self.maxtasksperchild,
    )

  def stop(self):
    self.pool.terminate()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.stop()

  def __get_divergences_np_based2(self, keys: OrderedSet[int]) -> np.ndarray:
    assert self.pool is not None
    # logger.debug(f"Using {thread_count} threads with {chunksize} chunks...")
    # logger.info("Calculating Kullback-Leibler divergences...")

    final_chunksize = get_chunksize(len(keys), self.n_jobs, self.chunksize, self.batches)
    log_mp_params(self.n_jobs, final_chunksize, self.maxtasksperchild, len(keys))

    result = np.zeros(shape=(len(keys)))
    # with Pool(
    #   processes=self.n_jobs,
    #   initializer=init_pool_np_based,
    #   initargs=(self.data, self.covered_array, self.target_dist),
    #   maxtasksperchild=self.maxtasksperchild,
    # ) as pool:
    iterator = self.pool.imap_unordered(
      get_divergence_for_utterance_np_based2, enumerate(keys), chunksize=final_chunksize
    )

    for index, kld in iterator:
      result[index] = kld

    return result

  def __iter__(self) -> Iterator[int]:
    return self

  def get_current_kld(self) -> float:
    current_distribution = get_distribution_array(self.covered_array)
    kld = entropy(current_distribution, self.target_dist)
    return kld

  def __next__(self) -> int:
    if len(self.available_data_keys_ordered) == 0:
      raise StopIteration()

    divergences = self.__get_divergences_np_based2(
      keys=self.available_data_keys_ordered,
    )

    # minima_indicies = np.argmin(divergences, axis=0)
    min_div = divergences.min()
    minima_indicies = np.flatnonzero(divergences == min_div)
    potential_keys = OrderedSet(
      self.available_data_keys_ordered[index] for index in minima_indicies
    )

    if len(potential_keys) > 1:
      logger = getLogger(__name__)
      logger.info(f"Found {len(potential_keys)} candidates for the current iteration.")

    selected_key = self.key_selector.select_key(potential_keys)
    assert 0 <= selected_key < len(self.data)
    self.covered_array += self.data[selected_key]
    self.available_data_keys_ordered.remove(selected_key)
    return selected_key


def get_uniform_distribution(ngrams: OrderedSet[NGramNr]) -> np.ndarray:
  assert len(ngrams) > 0
  # if len(ngrams) == 0:
  #   result = np.zeros(shape=(len(ngrams)), dtype=np.float32)
  #   return result
  distr = 1 / len(ngrams)
  result = np.full(shape=(len(ngrams)), dtype=np.float32, fill_value=distr)
  return result


def get_chunksize(data_count: int, n_jobs: int, chunksize: Optional[int], batches: Optional[int]) -> int:
  if batches is None:
    assert chunksize is not None
    assert chunksize > 0
    return chunksize

  if data_count == 0:
    return 1

  chunksize = math.ceil(data_count / n_jobs / batches)
  return chunksize


def get_empty_row_indicies(array: np.ndarray) -> np.ndarray:
  empty_entry_ids = np.where(~array.any(axis=1))[0]
  return empty_entry_ids


def remove_from_ordered_set_inplace(s: OrderedSet[int], indicies: Iterable[int]) -> None:
  for index in reversed(sorted(indicies)):
    assert 0 <= index < len(s)
    remove_entry = s[index]
    s.remove(remove_entry)


def get_all_ngrams_iterator(symbols: OrderedSet[str], n_gram: int) -> Iterator[NGram]:
  possible_ngrams = itertools.product(symbols, repeat=n_gram)
  return possible_ngrams


def generate_nummerated_ngrams(ngrams: Iterable[NGram]) -> Generator[Tuple[NGram, int], None, None]:
  nummerated_ngrams = ((k, i) for i, k in enumerate(ngrams))
  return nummerated_ngrams


def greedy_kld_uniform_ngrams(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> OrderedSet[int]:
  pass


class NGramExtractor():
  def __init__(self, data: Dict[int, Tuple[str, ...]]) -> None:
    self.data = data
    self.fitted = False

  def fit(self, consider_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]]) -> None:
    consider_keys_exist_in_data = consider_keys.issubset(self.data.keys())
    assert consider_keys_exist_in_data

    self.consider_keys = consider_keys
    self.n_gram = n_gram

    logger = getLogger(__name__)
    logger.info(f"Collecting data symbols...")
    data_symbols = get_unique_symbols(self.data, consider_keys)
    target_symbols = OrderedSet(sorted(data_symbols))
    if ignore_symbols is not None:
      target_symbols -= ignore_symbols

    logger.info(f"Calculating all possible {n_gram}-grams...")
    possible_ngrams = get_all_ngrams_iterator(target_symbols, n_gram)
    nummerated_ngrams = generate_nummerated_ngrams(possible_ngrams)
    self.ngram_nr_to_ngram: OrderedDictType[NGram, NGramNr] = OrderedDict(tqdm(nummerated_ngrams))
    self.all_ngram_nrs: OrderedSet[NGramNr] = OrderedSet(self.ngram_nr_to_ngram.values())
    self.fitted = True

  def predict(self, keys: Set[int], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> np.ndarray:
    assert self.fitted
    keys_are_subset_of_fitted_keys = keys.issubset(self.consider_keys)
    assert keys_are_subset_of_fitted_keys

    result = np.zeros(shape=(len(keys), len(self.all_ngram_nrs)), dtype=np.uint32)
    if len(keys) == 0:
      return result

    logger = getLogger(__name__)
    logger.info(f"Calculating {self.n_gram}-grams...")

    chunksize = get_chunksize(len(keys), n_jobs, chunksize, batches)
    log_mp_params(n_jobs, chunksize, maxtasksperchild, len(keys))

    method_proxy = partial(
      get_ngram_counts_from_data_entry,
      n=self.n_gram,
    )

    with Pool(
        processes=n_jobs,
        initializer=get_ngrams_counts_from_data_init_pool,
        initargs=(self.data, self.ngram_nr_to_ngram, self.all_ngram_nrs),
        maxtasksperchild=maxtasksperchild,
      ) as pool:
      with tqdm(total=len(keys)) as pbar:
        iterator = pool.imap_unordered(method_proxy, enumerate(keys), chunksize=chunksize)
        for index, counts in iterator:
          result[index] = counts
          pbar.update()

    return result


class KldIterator(Iterator[int]):
  def __init__(self, data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], key_selector: KeySelector, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
    self.data = data
    self.n_jobs = n_jobs
    self.maxtasksperchild = maxtasksperchild
    self.chunksize = chunksize
    self.batches = batches
    self.preselection_keys = preselection_keys
    self.select_from_keys = select_from_keys
    self.ignore_symbols = ignore_symbols
    self.n_gram = n_gram
    self.key_selector = key_selector
    self.__prepare()

  def start(self):
    self.core_iterator.start()

  def stop(self):
    self.core_iterator.stop()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.stop()

  # def get_ngrams_counts_from_data(self, keys: Set[int], ngram_nr_to_ngram: Dict[Tuple[str, ...], NGramNrs], ngram_nrs: OrderedSet[NGramNr]) -> np.ndarray:
  #   result = np.zeros(shape=(len(keys), len(ngram_nrs)), dtype=np.uint32)
  #   if len(data) == 0:
  #     return result

  #   chunksize = get_chunksize(len(keys), n_jobs, chunksize, batches)
  #   log_mp_params(n_jobs, chunksize, maxtasksperchild, len(keys))

  #   method_proxy = partial(
  #     get_ngram_counts_from_data_entry,
  #     n=n_gram,
  #   )

  #   with Pool(
  #       processes=n_jobs,
  #       initializer=get_ngrams_counts_from_data_init_pool,
  #       initargs=(data, ngram_nr_to_ngram, ngram_nrs),
  #       maxtasksperchild=maxtasksperchild,
  #     ) as pool:
  #     with tqdm(total=len(keys)) as pbar:
  #       iterator = pool.imap_unordered(method_proxy, enumerate(keys), chunksize=chunksize)
  #       for index, counts in iterator:
  #         result[index] = counts
  #         pbar.update()

  #   return result

  def __prepare(self):
    logger = getLogger(__name__)
    
    ngram_extractor = NGramExtractor(self.data)
    ngram_extractor.fit(self.select_from_keys | self.preselection_keys, self.n_gram, self.ignore_symbols)
    np_counts = ngram_extractor.predict(self.select_from_keys, self.n_jobs, self.maxtasksperchild, self.chunksize, self.batches)
    
    logger.info(f"Collecting data symbols...")
    data_symbols = get_unique_symbols(self.data, self.select_from_keys)
    logger.info(f"Collecting preselection symbols...")
    preselection_symbols = get_unique_symbols(self.data, self.preselection_keys)
    target_symbols = OrderedSet(sorted(data_symbols | preselection_symbols))
    if self.ignore_symbols is not None:
      target_symbols -= self.ignore_symbols

    logger.info(f"Calculating all possible {self.n_gram}-grams...")
    possible_ngrams = get_all_ngrams_iterator(target_symbols, self.n_gram)
    nummerated_ngrams = generate_nummerated_ngrams(possible_ngrams)
    ngram_nr_to_ngram: OrderedDictType[NGram, NGramNr] = OrderedDict(tqdm(nummerated_ngrams))
    all_ngram_nrs: OrderedSet[NGramNr] = OrderedSet(ngram_nr_to_ngram.values())

    logger.info(f"Calculating {self.n_gram}-grams...")
    np_counts = get_ngrams_counts_from_data(
      data=self.data,
      keys=self.select_from_keys,
      n_gram=self.n_gram,
      ngram_nr_to_ngram=ngram_nr_to_ngram,
      ngram_nrs=all_ngram_nrs,
      n_jobs=self.n_jobs,
      maxtasksperchild=self.maxtasksperchild,
      chunksize=self.chunksize,
      batches=self.batches,
    )

    # remove empty rows
    empty_row_indicies = get_empty_row_indicies(np_counts)
    remove_rows = len(empty_row_indicies) > 0
    if remove_rows:
      logger.info(
        f"Removing {len(empty_row_indicies)} empty row(s) out of {len(np_counts)} rows...")
      remove_from_ordered_set_inplace(self.select_from_keys, empty_row_indicies)
      np_counts = np.delete(np_counts, empty_row_indicies, axis=0)
      logger.info("Done.")
    del empty_row_indicies

    pre_np_counts = get_ngrams_counts_from_data(
      data=self.data,
      keys=self.preselection_keys,
      n_gram=self.n_gram,
      ngram_nr_to_ngram=ngram_nr_to_ngram,
      ngram_nrs=all_ngram_nrs,
      n_jobs=self.n_jobs,
      maxtasksperchild=self.maxtasksperchild,
      chunksize=self.chunksize,
      batches=self.batches,
    )

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
      f"Obtained {len(all_ngram_nrs)} {self.n_gram}-gram(s): {', '.join(ngrams_str)}.")

    if len(all_ngram_nrs) == 0:
      logger.info("Nothing to select.")
      return OrderedSet()

    target_distribution_array = get_uniform_distribution(all_ngram_nrs)
    logger.info(f"Target (uniform) distribution: {target_distribution_array[0]}")

    preselected_ngrams_count: int = np.sum(preselection_counts, axis=0)
    if preselected_ngrams_count > 0:
      preselection_distr = get_distribution_array(preselection_counts)
      preselection_kld = entropy(preselection_distr, target_distribution_array)
      logger.info(f"Preselection Kullback-Leibler divergence: {preselection_kld}")

    self.core_iterator = KldCoreIterator(
      data=np_counts,
      target_dist=target_distribution_array,
      chunksize=self.chunksize,
      maxtasksperchild=self.maxtasksperchild,
      n_jobs=self.n_jobs,
      batches=self.batches,
      preselection=preselection_counts,
      key_selector=self.key_selector,
    )

  def get_current_kld(self) -> float:
    return self.core_iterator.get_current_kld()

  def __iter__(self) -> Iterator[int]:
    return self

  def __next__(self) -> int:
    index = next(self.core_iterator)
    key = self.select_from_keys[index]
    return key


def greedy_kld_uniform_ngrams_seconds_with_preselection_perf(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], select_from_durations_s: Dict[int, float], seconds: float, duration_boundary: DurationBoundary, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> OrderedSet[int]:
  logger = getLogger(__name__)

  select_from_keys = get_duration_keys(select_from_durations_s, select_from_keys, duration_boundary)

  with KldIterator(
    data=data,
    select_from_keys=select_from_keys,
    preselection_keys=preselection_keys,
    key_selector=FirstKeySelector(),
    n_gram=n_gram,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
    ignore_symbols=ignore_symbols,
  ) as it:
    greedy_selected, enough_data_was_available = iterate_durations_dict(
      it, select_from_durations_s, seconds)

  if not enough_data_was_available:
    logger.warning(
      f"Aborted since no further data had been available!")

  logger.info(f"Obtained Kullback-Leibler divergence: {it.get_current_kld()}")
  result = OrderedSet(greedy_selected)
  return result

  logger.info(f"Collecting data symbols...")
  data_symbols = get_unique_symbols(data, select_from_keys)
  logger.info(f"Collecting preselection symbols...")
  preselection_symbols = get_unique_symbols(data, preselection_keys)
  target_symbols = OrderedSet(sorted(data_symbols | preselection_symbols))
  if ignore_symbols is not None:
    target_symbols -= ignore_symbols

  logger.info(f"Calculating all possible {n_gram}-grams...")
  possible_ngrams = get_all_ngrams_iterator(target_symbols, n_gram)
  nummerated_ngrams = generate_nummerated_ngrams(possible_ngrams)
  ngram_nr_to_ngram: OrderedDictType[NGram, NGramNr] = OrderedDict(tqdm(nummerated_ngrams))
  all_ngram_nrs: OrderedSet[NGramNr] = OrderedSet(ngram_nr_to_ngram.values())

  logger.info(f"Calculating {n_gram}-grams...")
  np_counts = get_ngrams_counts_from_data(
    data=data,
    keys=select_from_keys,
    n_gram=n_gram,
    ngram_nr_to_ngram=ngram_nr_to_ngram,
    ngram_nrs=all_ngram_nrs,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

  # remove empty rows
  empty_row_indicies = get_empty_row_indicies(np_counts)
  remove_rows = len(empty_row_indicies) > 0
  if remove_rows:
    logger.info(f"Removing {len(empty_row_indicies)} empty row(s) out of {len(np_counts)} rows...")
    remove_from_ordered_set_inplace(select_from_keys, empty_row_indicies)
    np_counts = np.delete(np_counts, empty_row_indicies, axis=0)
    logger.info("Done.")
  del empty_row_indicies

  # preselected_ngrams = tuple(ngram for ngrams in preselection_ngrams.values() for ngram in ngrams)
  # preselection_counts = get_count_array(preselected_ngrams, all_ngram_nrs)
  pre_np_counts = get_ngrams_counts_from_data(
    data=data,
    keys=preselection_keys,
    n_gram=n_gram,
    ngram_nr_to_ngram=ngram_nr_to_ngram,
    ngram_nrs=all_ngram_nrs,
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  )

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

  if len(all_ngram_nrs) == 0:
    logger.info("Nothing to select.")
    return OrderedSet()

  target_distribution_array = get_uniform_distribution(all_ngram_nrs)
  logger.info(f"Target (uniform) distribution: {target_distribution_array[0]}")

  preselected_ngrams_count: int = np.sum(preselection_counts, axis=0)
  if preselected_ngrams_count > 0:
    preselection_distr = get_distribution_array(preselection_counts)
    preselection_kld = entropy(preselection_distr, target_distribution_array)
    logger.info(f"Preselection Kullback-Leibler divergence: {preselection_kld}")

  until_values = np.zeros(shape=(len(np_counts)))
  for index, k in enumerate(select_from_keys):
    assert k in select_from_durations_s
    until_values[index] = select_from_durations_s[k]

  iterator = KldCoreIterator(
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

  logger.info(f"Obtained Kullback-Leibler divergence: {iterator.get_current_kld()}")

  result = OrderedSet([select_from_keys[index] for index in greedy_selected])

  return result


def iterate_durations_dict(iterator: Iterator[int], until_values: Dict[int, Union[int, float]], until_value: Union[int, float]) -> Tuple[List[int], bool]:
  iterated_values: List[int] = []
  enough_data_was_available = False
  max_until = sum(until_values)
  adjusted_until = round(min(until_value, max_until))
  current_total = 0.0
  with tqdm(total=adjusted_until, initial=round(current_total)) as progress_bar:
    for selected_key in iterator:
      assert selected_key in until_values
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


def get_unique_symbols(data: Dict[int, Tuple[str, ...]], keys: Set[int]) -> Set[str]:
  occurring_symbols = {symbol for key in tqdm(keys, total=len(keys)) for symbol in data[key]}
  return occurring_symbols

# region get_ngram_counts_from_data


def get_ngrams_counts_from_data(data: OrderedDictType[int, Tuple[str, ...]], keys: Set[int], n_gram: int, ngram_nr_to_ngram: Dict[Tuple[str, ...], NGramNrs], ngram_nrs: OrderedSet[NGramNr], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> np.ndarray:
  result = np.zeros(shape=(len(keys), len(ngram_nrs)), dtype=np.uint32)
  if len(data) == 0:
    return result

  chunksize = get_chunksize(len(keys), n_jobs, chunksize, batches)
  log_mp_params(n_jobs, chunksize, maxtasksperchild, len(keys))

  method_proxy = partial(
    get_ngram_counts_from_data_entry,
    n=n_gram,
  )

  with Pool(
      processes=n_jobs,
      initializer=get_ngrams_counts_from_data_init_pool,
      initargs=(data, ngram_nr_to_ngram, ngram_nrs),
      maxtasksperchild=maxtasksperchild,
    ) as pool:
    with tqdm(total=len(keys)) as pbar:
      iterator = pool.imap_unordered(method_proxy, enumerate(keys), chunksize=chunksize)
      for index, counts in iterator:
        result[index] = counts
        pbar.update()

  return result


process_ngram_nr_to_ngram: Dict[Tuple[str, ...], NGramNr] = None
process_data: OrderedDictType[int, Tuple[str, ...]] = None
process_ngram_nrs: OrderedSet[NGramNr] = None


def get_ngrams_counts_from_data_init_pool(data: OrderedDictType[int, Tuple[str, ...]], ngram_nr_to_ngram: Dict[Tuple[str, ...], NGramNr], ngram_nrs: OrderedSet[NGramNr]) -> None:
  global process_data
  global process_ngram_nr_to_ngram
  global process_ngram_nrs
  process_data = data
  process_ngram_nr_to_ngram = ngram_nr_to_ngram
  process_ngram_nrs = ngram_nrs


def get_ngram_counts_from_data_entry(index_key: Tuple[int, int], n: int) -> Tuple[int, np.ndarray]:
  global process_data
  global process_ngram_nr_to_ngram
  global process_ngram_nrs
  index, key = index_key
  symbols = process_data[key]

  ngram_nrs = (
    process_ngram_nr_to_ngram[ngram]
    for ngram in get_ngrams_generator(symbols, n)
    if ngram in process_ngram_nr_to_ngram
  )

  counts = Counter(ngram_nrs)
  res_tuple = tuple(
    counts.get(ngram_nr, 0)
    for ngram_nr in process_ngram_nrs
  )
  del counts

  result = np.array(res_tuple, dtype=np.uint32)
  del res_tuple

  return index, result


def get_count_array(ngrams: Iterable[NGramNr], target_symbols_ordered: OrderedSet[NGramNr]) -> np.ndarray:
  counts = Counter(ngrams)
  res_tuple = tuple(
    counts.get(ngram_nr, 0)
    for ngram_nr in target_symbols_ordered
  )

  result = np.array(res_tuple, dtype=np.uint32)
  return result

# endregion


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


def get_kld(dist: np.ndarray, target_dist: np.ndarray) -> float:
  none_of_targed_ngrams_exist = all(np.isnan(dist))
  if none_of_targed_ngrams_exist:
    return math.inf

  res = entropy(dist, target_dist)
  return res
