import math
from logging import getLogger
from multiprocessing import Pool, pool
from typing import Iterator, Optional, Tuple

import numpy as np
from ordered_set import OrderedSet
from scipy.stats import entropy
from text_selection.kld.distribution_factories import DistributionFactory
from text_selection.selection import KeySelector
from text_selection.utils import get_chunksize, log_mp_params


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
  del utterance_counts
  distr = get_distribution_array(counts)
  del counts
  kld = get_kld(distr, process_target_dist_np_based)
  del distr
  return index, kld


def get_kld(dist: np.ndarray, target_dist: np.ndarray) -> float:
  none_of_targed_ngrams_exist = all(np.isnan(dist))
  if none_of_targed_ngrams_exist:
    return math.inf

  res = entropy(dist, target_dist)
  return res


def get_distribution_array(counts: np.ndarray) -> np.ndarray:
  # assert all(x >= 0 for x in counts) # too slow
  sum_counts = np.sum(counts)
  new_dist = np.divide(counts, sum_counts)
  return new_dist
