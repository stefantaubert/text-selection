from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set

from ordered_set import OrderedSet
from text_selection.greedy_kld_applied import (
    greedy_kld_uniform_default, greedy_kld_uniform_count,
    greedy_kld_uniform_iterations, greedy_kld_uniform_seconds)
from text_selection.utils import get_filtered_ngrams


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


def greedy_kld_uniform_ngrams_count(data: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], chars: Dict[int, int], total_count: int) -> OrderedSet[int]:
  data_ngrams = get_filtered_ngrams(data, n_gram, ignore_symbols)
  return greedy_kld_uniform_count(
    data=data_ngrams,
    chars=chars,
    total_count=total_count,
  )
