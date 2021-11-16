import itertools
from collections import Counter, OrderedDict
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from typing import Dict, Generator, Iterable, Iterator, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

import numpy as np
from ordered_set import OrderedSet
from text_selection.utils import get_chunksize, log_mp_params
from tqdm import tqdm

NGramNr = int
NGramNrs = np.ndarray
NGram = Tuple[str, ...]


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


def get_unique_symbols(data: Dict[int, Tuple[str, ...]], keys: Set[int]) -> Set[str]:
  occurring_symbols = {symbol for key in tqdm(keys, total=len(keys)) for symbol in data[key]}
  return occurring_symbols


def get_all_ngrams_iterator(symbols: OrderedSet[str], n_gram: int) -> Iterator[NGram]:
  possible_ngrams = itertools.product(symbols, repeat=n_gram)
  return possible_ngrams


def generate_nummerated_ngrams(ngrams: Iterable[NGram]) -> Generator[Tuple[NGram, int], None, None]:
  nummerated_ngrams = ((k, i) for i, k in enumerate(ngrams))
  return nummerated_ngrams


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


def get_ngrams_generator(sentence_symbols: List[str], n: int) -> Generator[Tuple[str, ...], None, None]:
  # TODO: import from text-utils
  if n < 1:
    raise Exception()

  result = (
    tuple(sentence_symbols[i:i + n])
    for i in range(len(sentence_symbols) - n + 1)
  )
  return result
