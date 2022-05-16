from collections import Counter, OrderedDict
from functools import partial
from itertools import chain
from logging import Logger, getLogger
from multiprocessing import Pool
from time import perf_counter
from typing import Dict, List, Optional
from typing import Set, Tuple

import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.helper import split_adv
from text_selection_core.types import Lines, Subset

SymbolIndex = int
SymbolCounts = np.ndarray


def xtqdm(x, desc=None, unit=None, total=None):
  yield from x


def get_symbols_from_lines(lines: Lines, subset: Subset, ssep: str) -> Set[str]:
  symbols = {
    symbol
    for line_nr in tqdm(subset, desc="Getting unique symbols", unit=" line(s)")
    for symbol in split_adv(lines[line_nr], ssep)
  }
  return symbols


def get_array(lines: Lines, subset: Subset, ssep: str, logger: Logger):
  start = perf_counter()
  # 1mio -> 8.4
  # 10mio -> 84.94535316200927s
  # get_array_v1(lines, subset, ssep, logger)
  # 1mio -> 10.079393691034056s
  # 10mio -> 92.748770733946s
  # get_array_v2(lines, subset, ssep, logger)  # 10.079393691034056s
  result = get_array_mp(lines, subset, ssep, logger, 1_000_000, 16, None)
  duration = perf_counter() - start
  logger = getLogger()
  logger.info(f"Duration: {duration}s")
  return result


def get_chunks(keys: List[str], chunk_size: Optional[int]) -> List[List[str]]:
  if chunk_size is None:
    chunk_size = len(keys)
  chunked_list = list(keys[i: i + chunk_size] for i in range(0, len(keys), chunk_size))
  return chunked_list


def get_array_mp(lines: Lines, subset: Subset, ssep: str, logger: Logger, chunksize: int, n_jobs: int, maxtasksperchild: Optional[int]):
  subset_chunks = get_chunks(subset, chunksize)
  method = partial(
    get_array_v1_mp,
    ssep=ssep,
    logger=logger,
  )

  logger.debug(f"# Lines: {len(lines)}")
  logger.debug(f"# Subset lines: {len(subset)}")
  logger.debug(f"Chunksize: {chunksize}")
  logger.debug(f"Chunks: {len(subset_chunks)}")
  logger.debug(f"Maxtask: {maxtasksperchild}")
  logger.debug(f"Jobs: {n_jobs}")

  n_jobs = min(n_jobs, len(subset_chunks))
  logger.debug(f"Jobs (final): {n_jobs}")

  with Pool(
    processes=n_jobs,
    maxtasksperchild=maxtasksperchild,
    initializer=init_mp,
    initargs=(lines,),
  ) as pool:
    arrays = dict(tqdm(pool.imap_unordered(
        method, enumerate(subset_chunks), chunksize=1
    ), total=len(subset_chunks), desc="Generating data"))
  arrays = sorted(arrays.items(), key=lambda kv: kv[0])
  arrays = list(v for k, v in arrays)
  arrays, symbols = unify_arrays(arrays)
  return arrays, symbols


def init_mp(lines: Lines):
  global process_lines
  process_lines = lines


def get_array_v1_mp(i_subset: Tuple[int, Subset], ssep: str, logger: Logger):
  global process_lines
  i, subset = i_subset
  result = get_array_v1(process_lines, subset, ssep, logger)
  return i, result


def unify_arrays(arrays_symbols: List[Tuple[np.ndarray, OrderedSet[str]]]) -> Tuple[np.ndarray, OrderedSet[str]]:
  all_symbols = OrderedSet({
    symbol
    for _, symbols in arrays_symbols
    for symbol in symbols
  })

  result = None
  symbols: OrderedSet[str]

  for array, symbols in tqdm(arrays_symbols, desc="Merging results"):
    missing_symbols = all_symbols - symbols
    if len(missing_symbols) > 0:
      new_cols = np.zeros((len(array), len(missing_symbols)), dtype=np.uint32)
      array = np.append(array, new_cols, axis=1)
      symbols.update(missing_symbols)

    assert len(symbols) == len(all_symbols)
    symbols_mapping = [
      symbols.index(s)
      for s in all_symbols
    ]

    array = array[:, symbols_mapping]
    # TODO extract this to upper function and replace results in list instead
    if result is None:
      result = array
    else:
      result = np.append(result, array, axis=0)
  return result, all_symbols


def merge_arrays_v1(array_keys: List[Tuple[np.ndarray, OrderedSet[str]]]):
  result_data: np.ndarray = None
  result_symbols: Dict = None
  for current_array, current_symbols in tqdm(array_keys, desc="Merging results"):
    if result_data is None:
      result_data = current_array
      result_symbols = OrderedDict(
        sorted(current_symbols.items(), key=lambda kv: kv[1], reverse=False))
      assert list(result_symbols.values()) == list(range(len(current_symbols)))
    else:
      current_symbols = OrderedDict(
        sorted(current_symbols.items(), key=lambda kv: kv[1], reverse=False))
      new_symbols = current_symbols.keys() - result_symbols.keys()
      if len(new_symbols) > 0:
        for new_symbol in new_symbols:
          result_symbols[new_symbol] = len(result_symbols)

      missing_symbols = result_symbols.keys() - current_symbols.keys()
      if len(missing_symbols) > 0:
        for missing_symbol in missing_symbols:
          current_symbols[missing_symbol] = len(current_symbols)
      assert len(current_symbols) == len(result_symbols)
      assert list(current_symbols.values()) == list(result_symbols.values())
      new_col = np.zeros((len(result_data), len(new_symbols)), dtype=np.uint32)
      result_data = np.append(result_data, new_col, axis=1)
      mapping = [
        result_symbols[symbol]
        for symbol, index in current_symbols.items()
      ]

      current_array = current_array[:, mapping]
      result_data = np.append(result_data, current_array, axis=0)

  return result_data, result_symbols


def get_array_v1(lines: Lines, subset: Subset, ssep: str, logger: Logger):
  counters = []
  for line_nr in xtqdm(subset, desc="Calculating counts", unit=" line(s)"):
    line_counts = Counter(split_adv(lines[line_nr], ssep))
    counters.append(line_counts)

  #logger.info("Getting symbols")
  symbols = OrderedSet(set(chain(*counters)))
  # logger.info("Done.")

  if "" in symbols:
    symbols.remove("")

  symbols_indices = dict((s, i) for i, s in enumerate(symbols))

  result = np.zeros((len(counters), len(symbols)), dtype=np.uint32)

  for line_index, c in enumerate(xtqdm(counters)):
    for symbol, count in c.items():
      if symbol in symbols_indices:
        symbol_index = symbols_indices[symbol]
        result[line_index, symbol_index] = count

  return result, symbols

  # counter += line_counts
  x = chain(*counters)
  counter = Counter(x)
  symbols = get_symbols_from_lines(lines, subset, ssep)
  if "" in symbols:
    symbols.remove("")

  numerated_symbols = dict((k, i) for i, k in enumerate(symbols))

  # logger.info(len(numerated_symbols))


def get_array_v2(lines: Lines, subset: Subset, ssep: str, logger: Logger):
  result = None
  symbols_indicies = {}

  for line_nr in tqdm(subset, desc="Calculating counts", unit=" line(s)"):
    line_counts = Counter(lines[line_nr].split(ssep))
    if result is None:
      result = np.zeros((len(subset), len(line_counts)), dtype=np.uint32)
      symbols_indicies = dict((s, i) for i, s in enumerate(line_counts))
    else:
      new_symbols = line_counts.keys() - symbols_indicies.keys()
      # new_symbols = set(line_counts.keys()).difference(symbols_indicies.keys())
      if len(new_symbols) > 0:
        new_col = np.zeros((len(subset), len(new_symbols)), dtype=np.uint32)
        result = np.append(result, new_col, axis=1)
        for new_symbol in new_symbols:
          symbols_indicies[new_symbol] = len(symbols_indicies)

    for symbol, count in line_counts.items():
      symbol_index = symbols_indicies[symbol]
      result[line_nr, symbol_index] = count
  return result, symbols_indicies
