from collections import Counter
from dataclasses import dataclass
from logging import getLogger
from typing import Generator, Iterable, Set, Tuple

from ordered_set import OrderedSet
from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (DataId, DataSymbols, Subset,
                                       get_subsets_ids, item_to_text,
                                       move_ids_to_subset)
from text_utils import Symbol, symbols_split_iterable, symbols_strip
from tqdm import tqdm


@dataclass()
class WordsCountFilterParameters():
  symbols: DataSymbols
  word_sep: Symbol
  from_count_incl: int
  to_count_excl: int
  ignore_case: bool
  trim_symbols: Set[Symbol]


def filter_words_with_frequencies(default_params: SelectionDefaultParameters, params: WordsCountFilterParameters) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  select_from = ((data_id, item_to_text(params.data_symbols[data_id]))
                 for data_id in get_subsets_ids(default_params.dataset, default_params. from_subset_names))
  counter = get_counter(select_from, params.word_sep, params.trim_symbols, params.ignore_case)

  select_from = ((data_id, item_to_text(params.data_symbols[data_id]))
                 for data_id in get_subsets_ids(default_params.dataset, default_params. from_subset_names))
  items = get_matching_items(select_from, params.word_sep, params.trim_symbols,
                             params.ignore_case, params.from_count_incl, params.to_count_excl, counter)
  result: Subset = OrderedSet(items)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(result)} Id's.")
    move_ids_to_subset(default_params.dataset, result, default_params.to_subset_name)
    changed_anything = True
  return None, changed_anything


def get_counter(items: Iterable[Tuple[DataId, str]], word_sep: Symbol, trim_symbols: Set[Symbol], ignore_case: bool) -> Generator[DataId, None, None]:
  assert len(word_sep) > 0

  words = (
      word
      for _, sentence in items
      for word in symbols_split_iterable(sentence, {word_sep})
      if word != ""
  )

  if len(trim_symbols) > 0:
    words = (symbols_strip(word, trim_symbols) for word in words)

  words = (''.join(word) for word in words)

  if ignore_case:
    words = (word.lower() for word in words)

  logger = getLogger(__name__)
  logger.debug("Counting words...")
  counter = Counter(tqdm(words))

  return counter


def get_matching_items(items: Iterable[Tuple[DataId, str]], word_sep: Symbol, trim_symbols: Set[Symbol], ignore_case: bool, from_count_incl: int, to_count_excl: int, counter: Counter) -> Generator[DataId, None, None]:
  assert len(word_sep) > 0

  items = (
      (data_id, (word for word in symbols_split_iterable(sentence, {word_sep}) if word != ""))
      for data_id, sentence in items
  )

  if len(trim_symbols) > 0:
    items = (
      (data_id, (symbols_strip(word, trim_symbols) for word in words))
      for data_id, words in items
    )

  items = (
    (data_id, (''.join(word) for word in words))
    for data_id, words in items
  )

  if ignore_case:
    items = (
      (data_id, (word.lower() for word in words))
      for data_id, words in items
    )

  items_in_range = (
    (data_id, all(
      from_count_incl <= counter[word] < to_count_excl for word in words
    ))
    for data_id, words in items
  )

  for data_id, all_words_in_range in items_in_range:
    if all_words_in_range:
      yield data_id
