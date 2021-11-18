import math
from logging import getLogger
from multiprocessing import Pool, pool
from typing import Iterator, Optional, Tuple

import numpy as np
from ordered_set import OrderedSet
from scipy.stats import entropy
from text_selection.selection import KeySelector
from text_selection.utils import get_chunksize, log_mp_params


def get_uniform_weights(count: int) -> np.ndarray:
  result = np.ones(shape=(count), dtype=np.uint16)
  return result


def get_distribution_from_weights(weights: np.ndarray) -> np.ndarray:
  assert len(weights.shape) == 1
  summed_weights = np.sum(weights, axis=0)
  assert summed_weights > 0
  probabilities = np.divide(weights, summed_weights)
  return probabilities


class KldIterator(Iterator[int]):
  def __init__(self, data: np.ndarray, data_indicies: OrderedSet[int], weights: np.ndarray, preselection: np.ndarray, key_selector: KeySelector, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
    super().__init__()
    self._data = data
    self._key_selector = key_selector
    # defines the order for what the selection is based on
    self.__available_data_keys_ordered = data_indicies
    self.__covered_array = preselection.copy()
    self.__n_jobs = n_jobs
    self.__maxtasksperchild = maxtasksperchild
    self.__chunksize = chunksize
    self.__batches = batches
    self.__pool: pool.Pool = None
    self.__target_dist = get_distribution_from_weights(weights)
    self._previous_kld: Optional[float] = None
    self.__current_kld: float = math.inf

    covered_distribution = get_distribution_array(self.__covered_array)
    self.__current_kld = get_kld(covered_distribution, self.__target_dist)
    del covered_distribution

  def start(self):
    assert self.__pool is None
    self.__pool = Pool(
      processes=self.__n_jobs,
      initializer=init_pool_np_based,
      initargs=(self._data, self.__covered_array, self.__target_dist),
      maxtasksperchild=self.__maxtasksperchild,
    )

  def stop(self):
    assert self.__pool is not None
    self.__pool.terminate()
    self.__pool = None

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.stop()

  def __get_divergences(self, keys: OrderedSet[int]) -> np.ndarray:
    assert self.__pool is not None
    final_chunksize = get_chunksize(len(keys), self.__n_jobs, self.__chunksize, self.__batches)
    log_mp_params(self.__n_jobs, final_chunksize, self.__maxtasksperchild, len(keys))
    iterator = self.__pool.imap_unordered(
      get_divergence_for_utterance_np_based2, enumerate(keys),
      chunksize=final_chunksize
    )

    result = np.zeros(shape=(len(keys)), dtype=np.float64)
    for index, kld in iterator:
      result[index] = kld

    return result

  def __iter__(self) -> Iterator[int]:
    return self

  @property
  def previous_kld(self) -> Optional[float]:
    return self._previous_kld

  @property
  def current_kld(self) -> float:
    return self.__current_kld

  @property
  def target_distribution(self) -> np.ndarray:
    return self.__target_dist

  def __next__(self) -> int:
    if len(self.__available_data_keys_ordered) == 0:
      raise StopIteration()

    divergences = self.__get_divergences(
      keys=self.__available_data_keys_ordered,
    )

    min_div = divergences.min()
    minima_indicies = np.flatnonzero(divergences == min_div)
    potential_keys = OrderedSet(
      self.__available_data_keys_ordered[index] for index in minima_indicies
    )

    if len(potential_keys) > 1:
      logger = getLogger(__name__)
      logger.info(f"Found {len(potential_keys)} candidates for the current iteration.")

    selected_key = self._key_selector.select_key(potential_keys)
    assert 0 <= selected_key < len(self._data)
    self.__covered_array += self._data[selected_key]
    self.__available_data_keys_ordered.remove(selected_key)
    self._previous_kld = self.__current_kld
    self.__current_kld = min_div
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


def get_divergence_for_utterance(key: int, data: np.ndarray, covered_counts: np.ndarray, target_dist: np.ndarray) -> float:
  assert 0 <= key < len(data)
  utterance_counts = data[key]
  counts = covered_counts + utterance_counts
  del utterance_counts
  distr = get_distribution_array(counts)
  del counts
  kld = get_kld(distr, target_dist)
  del distr
  return kld


def get_divergence_for_utterance_np_based2(index_key: Tuple[int, int]) -> Tuple[int, float]:
  # pylint: disable=global-variable-not-assigned
  global process_data_np_based
  global process_covered_counts_np_based
  global process_target_dist_np_based
  index, key = index_key
  result = get_divergence_for_utterance(
    key=key,
    covered_counts=process_covered_counts_np_based,
    data=process_data_np_based,
    target_dist=process_target_dist_np_based,
  )

  del key

  return index, result


def get_kld(dist: np.ndarray, target_dist: np.ndarray) -> float:
  assert len(dist.shape) == len(target_dist.shape) == 1
  assert len(dist) == len(target_dist)
  # dist and target_dist must not be all zero or have negative values
  # target_dist must not be all zero
  assert len(dist) == 0 or len(dist[dist == 0]) < len(dist)
  assert len(dist) == 0 or len(target_dist[target_dist == 0]) < len(target_dist)
  assert len(dist[dist < 0]) == 0
  assert len(target_dist[target_dist < 0]) == 0

  res = entropy(dist, target_dist, axis=0)
  if np.isnan(res):
    del res
    return np.inf

  return res


def get_distribution_array(counts: np.ndarray) -> np.ndarray:
  assert len(counts.shape) == 1
  assert len(counts[counts < 0]) == 0
  sum_counts = np.sum(counts)
  new_dist = np.divide(counts, sum_counts)
  del sum_counts
  return new_dist
