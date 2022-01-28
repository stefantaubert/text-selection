from text_selection_core.types import (DataIds, DataSymbols, DataWeights,
                                       item_to_symbols, item_to_text)
from text_utils import Symbol, get_words


def get_uniform_weights(ids: DataIds) -> DataWeights:
  result = dict.fromkeys(ids, 1)
  return result


def get_character_count_weights(data_symbols: DataSymbols) -> DataWeights:
  result = {
    data_id: len(item_to_symbols(symbols_str))
    for data_id, symbols_str in data_symbols
  }

  return result


def get_word_count_weights(data_symbols: DataSymbols, word_sep: Symbol) -> DataWeights:
  texts = (
    (data_id, item_to_text(symbols_str))
    for data_id, symbols_str in data_symbols.items()
  )

  words_counts = (
    (data_id, sum(1 for word in item.split(word_sep) if word != ""))
    for data_id, item in texts
  )

  result = dict(words_counts)

  return result


def divide_weights_inplace(weights: DataWeights, divide_by: float) -> None:
  assert divide_by > 0
  for data_id in weights:
    weights[data_id] /= divide_by
