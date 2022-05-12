from text_selection_core.types import DataWeights, LineNrs, Lines


def get_uniform_weights(ids: LineNrs) -> DataWeights:
  result = dict.fromkeys(ids, 1)
  return result


def get_character_count_weights(data_symbols: Lines) -> DataWeights:
  result = {
    data_id: len(symbols_str)
    for data_id, symbols_str in data_symbols.items()
  }

  return result


def get_word_count_weights(data_symbols: Lines, word_sep: str) -> DataWeights:
  texts = (
    (data_id, symbols_str)
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
