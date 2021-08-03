from collections import OrderedDict
from math import inf
from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set

from ordered_set import OrderedSet

from text_selection.greedy_kld_applied import (
    greedy_kld_uniform_count, greedy_kld_uniform_default,
    greedy_kld_uniform_iterations, greedy_kld_uniform_parts,
    greedy_kld_uniform_seconds, greedy_kld_uniform_seconds_with_preselection)
from text_selection.selection import SelectionMode
from text_selection.utils import (DurationBoundary, filter_data_durations,
                                  get_filtered_ngrams)


def greedy_kld_uniform_ngrams_parts(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], parts_count: int, take_per_part: int) -> OrderedSet[int]:
  data_ngrams = get_filtered_ngrams(data, n_gram, ignore_symbols)
  lengths = OrderedDict({k: len(v) for k, v in data.items()})
  return greedy_kld_uniform_parts(
    data=data_ngrams,
    take_per_part=take_per_part,
    parts_count=parts_count,
    lengths=lengths,
  )


def greedy_kld_uniform_ngrams_default(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]]) -> OrderedSet[int]:
  data_ngrams = get_filtered_ngrams(data, n_gram, ignore_symbols)
  return greedy_kld_uniform_default(
    data=data_ngrams,
  )


def greedy_kld_uniform_ngrams_iterations(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], iterations: int) -> OrderedSet[int]:
  data_ngrams = get_filtered_ngrams(data, n_gram, ignore_symbols)
  return greedy_kld_uniform_iterations(
    data=data_ngrams,
    iterations=iterations,
  )


def greedy_kld_uniform_ngrams_seconds(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], durations_s: Dict[int, float], seconds: float) -> OrderedSet[int]:
  data_ngrams = get_filtered_ngrams(data, n_gram, ignore_symbols)
  return greedy_kld_uniform_seconds(
    data=data_ngrams,
    durations_s=durations_s,
    seconds=seconds,
  )


def greedy_kld_uniform_ngrams_seconds_with_preselection(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], durations_s: Dict[int, float], seconds: float, preselection: OrderedDictType[int, List[str]], duration_boundary: DurationBoundary = (0, inf), mp: bool = True) -> OrderedSet[int]:
  data_ngrams = get_filtered_ngrams(data, n_gram, ignore_symbols)
  data_ngrams = filter_data_durations(data_ngrams, durations_s, duration_boundary)
  preselection_ngrams = get_filtered_ngrams(preselection, n_gram, ignore_symbols)

  return greedy_kld_uniform_seconds_with_preselection(
    data=data_ngrams,
    durations_s=durations_s,
    seconds=seconds,
    preselection=preselection_ngrams,
    mp=mp,
  )


def greedy_kld_uniform_ngrams_count(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], chars: Dict[int, int], total_count: int) -> OrderedSet[int]:
  data_ngrams = get_filtered_ngrams(data, n_gram, ignore_symbols)
  return greedy_kld_uniform_count(
    data=data_ngrams,
    chars=chars,
    total_count=total_count,
  )
