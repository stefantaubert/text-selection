from dataclasses import dataclass
from logging import getLogger
from typing import Generator, Iterable, Literal, Set, Tuple

import enchant
from ordered_set import OrderedSet
from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (DataId, DataSymbols, Subset,
                                       get_subsets_ids, item_to_symbols,
                                       move_ids_to_subset)
from text_utils import Symbol, Symbols, symbols_split_iterable, symbols_strip

enchant_en_US = Literal["en_US"]


@dataclass()
class Parameters():
  symbols: DataSymbols
  word_sep: Symbol
  trim_symbols: Set[Symbol]
  language: Literal[enchant_en_US]


def filter_sentences_containing_only_known_words(default_params: SelectionDefaultParameters, params: Parameters) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  select_from = ((data_id, item_to_symbols(params.data_symbols[data_id]))
                 for data_id in get_subsets_ids(default_params.dataset, default_params. from_subset_names))

  assert params.language == enchant_en_US
  lexicon = enchant.Dict(params.language)

  items = get_matching_items(select_from, params.word_sep, params.trim_symbols, lexicon)
  result: Subset = OrderedSet(items)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(result)} Id's.")
    move_ids_to_subset(default_params.dataset, result, default_params.to_subset_name)
    changed_anything = True
  return None, changed_anything


def get_matching_items(items: Iterable[Tuple[DataId, Symbols]], word_sep: Symbol, trim_symbols: Set[Symbol], lexicon: enchant.Dict) -> Generator[DataId, None, None]:
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

  # Note: all on first false occurrence: https://github.com/python/cpython/blob/b9d8980d89bfaa4bf16d60f0488adcc9d2cbf5ef/Doc/library/functions.rst#id7
  items = (
    (data_id, all(lexicon.check(word) for word in words))
    for data_id, words in items
  )

  for data_id, contains_only_known_words in items:
    if contains_only_known_words:
      yield data_id
