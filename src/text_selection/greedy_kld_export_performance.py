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


class DistributionFactory():
  def get(self, count: int) -> np.ndarray:
    raise NotImplementedError()


class UniformDistributionFactory(DistributionFactory):
  def get(self, count: int) -> np.ndarray:
    if count == 0:
      result = np.zeros(shape=(0), dtype=np.float32)
      return result
    distr = 1 / count
    result = np.full(shape=(count), dtype=np.float64, fill_value=1)
    result = np.divide(result, count)
    return result


class KldCoreIterator(Iterator[int]):
  def __init__(self, data: np.ndarray, data_indicies: OrderedSet[int], distribution_factory: DistributionFactory, preselection: np.ndarray, key_selector: KeySelector, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
    super().__init__()
    self.key_selector = key_selector
    # defines the order for what the selection is based on
    # self.available_data_keys_ordered = OrderedSet(range(len(data)))
    self.available_data_keys_ordered = data_indicies
    self.data = data
    self.covered_array = preselection.copy()
    self.n_jobs = n_jobs
    self.maxtasksperchild = maxtasksperchild
    self.chunksize = chunksize
    self.batches = batches
    self.pool: pool.Pool = None
    self.target_dist = distribution_factory.get(data.shape[1])
    self.last_selected_key: Optional[int] = None

  def start(self):
    self.pool = Pool(
      processes=self.n_jobs,
      initializer=init_pool_np_based,
      initargs=(self.data, self.covered_array, self.target_dist),
      maxtasksperchild=self.maxtasksperchild,
    )

  def get_target_distribution(self) -> np.ndarray:
    return self.target_dist

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
    # preselected_ngrams_count: int = np.sum(self.covered_array, axis=0)
    # if preselected_ngrams_count > 0:
    current_distribution = get_distribution_array(self.covered_array)
    kld = get_kld(current_distribution, self.target_dist)
    return kld

  def __next__(self) -> int:
    if self.last_selected_key is not None:
      self.covered_array += self.data[self.last_selected_key]
      self.available_data_keys_ordered.remove(self.last_selected_key)

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
    self.last_selected_key = selected_key
    # self.covered_array += self.data[selected_key]
    # self.available_data_keys_ordered.remove(selected_key)
    return selected_key


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
  def __init__(self, data: Dict[int, Tuple[str, ...]], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
    self.data = data
    self.fitted = False
    self.n_jobs = n_jobs
    self.maxtasksperchild = maxtasksperchild
    self.chunksize = chunksize
    self.batches = batches

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
    self.all_ngrams: OrderedSet[NGram] = OrderedSet(self.ngram_nr_to_ngram.keys())

    ngrams_str = [
      f"\"{''.join(n_gram)}\"" for n_gram in self.all_ngrams]

    logger.info(
      f"Obtained {len(self.all_ngrams)} different {self.n_gram}-gram(s): {', '.join(ngrams_str)}.")
    self.fitted = True

  @property
  def fitted_ngrams(self) -> OrderedSet[NGram]:
    assert self.fitted
    return self.all_ngrams

  def predict(self, keys: Set[int]) -> np.ndarray:
    assert self.fitted
    keys_are_subset_of_fitted_keys = keys.issubset(self.consider_keys)
    assert keys_are_subset_of_fitted_keys

    result = np.zeros(shape=(len(keys), len(self.all_ngram_nrs)), dtype=np.uint32)
    if len(keys) == 0:
      return result

    logger = getLogger(__name__)
    logger.info(f"Calculating {self.n_gram}-grams...")

    final_chunksize = get_chunksize(len(keys), self.n_jobs, self.chunksize, self.batches)
    log_mp_params(self.n_jobs, final_chunksize, self.maxtasksperchild, len(keys))

    method_proxy = partial(
      get_ngram_counts_from_data_entry,
      n=self.n_gram,
    )

    with Pool(
        processes=self.n_jobs,
        initializer=get_ngrams_counts_from_data_init_pool,
        initargs=(self.data, self.ngram_nr_to_ngram, self.all_ngram_nrs),
        maxtasksperchild=self.maxtasksperchild,
      ) as pool:
      with tqdm(total=len(keys)) as pbar:
        iterator = pool.imap_unordered(method_proxy, enumerate(keys), chunksize=final_chunksize)
        for index, counts in iterator:
          result[index] = counts
          pbar.update()

    return result


class OptimizedKldIterator(KldCoreIterator):
  def __init__(self, data: np.ndarray, data_indicies: OrderedSet[int], preselection: np.ndarray, distribution_factory: DistributionFactory, key_selector: KeySelector, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
    logger = getLogger(__name__)
    logger.info("Copy data and preselection")
    super().__init__(
      data=data.copy(),
      preselection=preselection,
      data_indicies=data_indicies,
      distribution_factory=distribution_factory,
      batches=batches,
      chunksize=chunksize,
      key_selector=key_selector,
      maxtasksperchild=maxtasksperchild,
      n_jobs=n_jobs,
    )
    logger.info("Done")

    # remove empty rows
    empty_row_indicies = get_empty_row_indicies(self.data)
    remove_rows = len(empty_row_indicies) > 0
    if remove_rows:
      logger.info(
        f"Removing {len(empty_row_indicies)} empty row(s) out of {len(self.data)} rows...")
      remove_from_ordered_set_inplace(self.available_data_keys_ordered, empty_row_indicies)
      # self.data = np.delete(self.data, empty_row_indicies, axis=0)
      logger.info("Done.")
    self.available_empty_row_indicies = OrderedSet(empty_row_indicies)
    del empty_row_indicies
    # mapping = {index: key for index, key in enumerate(self.select_from_keys)}
    # self.mapping_iterator = MappingIterator(self, mapping=mapping)

    # remove empty columns, can only occur if len(symbols in utterance) = n_gram - 1
    data_counts: NDArray = np.sum(self.data, axis=0)
    all_counts: NDArray = data_counts + self.covered_array
    remove_ngram_indicies = np.where(all_counts == 0)[0]
    if len(remove_ngram_indicies) > 0:
      logger.info(
        f"Removing {len(remove_ngram_indicies)} out of {self.data.shape[1]} columns...")
      self.data: np.ndarray = np.delete(self.data, remove_ngram_indicies, axis=1)
      self.covered_array: np.ndarray = np.delete(self.covered_array, remove_ngram_indicies, axis=0)
      self.target_dist = distribution_factory.get(self.data.shape[1])
      logger.info("Done.")

  def __iter__(self) -> Iterator[int]:
    return self

  def __next__(self) -> int:
    # index = super().__next__()
    # key = self.select_from_keys[index]
    # return key
    try:
      return super().__next__()
    except StopIteration:
      if len(self.available_empty_row_indicies) > 0:
        selected_key = self.key_selector.select_key(self.available_empty_row_indicies)
        assert 0 <= selected_key < len(self.data)
        assert np.sum(self.data[selected_key], axis=0) == 0
        self.available_empty_row_indicies.remove(selected_key)
        return selected_key
      else:
        raise StopIteration()


class MappingIterator(Iterator[int]):
  def __init__(self, iterator: Iterator[int], mapping: Dict[int, int]) -> None:
    super().__init__()
    self.iterator = iterator
    self.mapping = mapping

  def __iter__(self) -> Iterator[int]:
    return self

  def __next__(self) -> int:
    result = next(self.iterator)
    assert result in self.mapping
    mapped_result = self.mapping[result]
    return mapped_result


def greedy_kld_uniform_ngrams_seconds_with_preselection_perf(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], select_from_durations_s: Dict[int, float], seconds: float, duration_boundary: DurationBoundary, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
  logger = getLogger(__name__)

  select_from_keys = get_duration_keys(select_from_durations_s, select_from_keys, duration_boundary)

  ngram_extractor = NGramExtractor(data, n_jobs, maxtasksperchild, chunksize, batches)
  ngram_extractor.fit(select_from_keys | preselection_keys, n_gram, ignore_symbols)
  all_data_counts = ngram_extractor.predict(select_from_keys)
  all_preselected_counts = ngram_extractor.predict(preselection_keys)
  summed_preselection_counts: NDArray = np.sum(all_preselected_counts, axis=0)
  del all_preselected_counts
  del ngram_extractor

  with OptimizedKldIterator(
    data=all_data_counts,
    preselection=summed_preselection_counts,
    data_indicies=OrderedSet(range(len(all_data_counts))),
    key_selector=FirstKeySelector(),
    distribution_factory=UniformDistributionFactory(),
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  ) as iterator:
    logger.info(f"Target (uniform) distribution: {iterator.get_target_distribution()[0]}")
    logger.info(f"Initial Kullback-Leibler divergence: {iterator.get_current_kld()}")
    key_index_mapping = {index: key for index, key in enumerate(select_from_keys)}
    mapping_iterator = MappingIterator(iterator, key_index_mapping)
    greedy_selected, enough_data_was_available = iterate_durations_dict(
      mapping_iterator, select_from_durations_s, seconds)

    if not enough_data_was_available:
      logger.warning(
        f"Aborted since no further data had been available!")

    logger.info(f"Final Kullback-Leibler divergence: {iterator.get_current_kld()}")

  result = OrderedSet(greedy_selected)
  return result


def greedy_kld_uniform_ngrams_seconds_with_preselection_perf_old(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], select_from_durations_s: Dict[int, float], seconds: float, duration_boundary: DurationBoundary, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> OrderedSet[int]:
  logger = getLogger(__name__)

  select_from_keys = get_duration_keys(select_from_durations_s, select_from_keys, duration_boundary)

  with OptimizedKldIterator(
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
