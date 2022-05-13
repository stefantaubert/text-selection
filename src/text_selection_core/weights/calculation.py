from tqdm import tqdm

from text_selection_core.types import DataWeights, Lines


def get_uniform_weights(line_nrs: range) -> DataWeights:
  result = dict.fromkeys(line_nrs, 1)
  return result


def get_word_count_weights(lines: Lines, sep: str) -> DataWeights:
  texts = (
    (data_id, line)
    for data_id, line in enumerate(lines, start=1)
  )

  if sep == "":
    words_counts = (
      (data_id, len(item))
      for data_id, item in texts
    )
  else:
    words_counts = (
      (data_id, sum(1 for word in item.split(sep) if word != ""))
      for data_id, item in texts
    )

  result = dict(tqdm(words_counts, desc="Getting counts", unit=" line(s)", total=len(lines)))

  return result


def divide_weights_inplace(weights: DataWeights, divide_by: float) -> None:
  assert divide_by > 0
  for data_id in weights:
    weights[data_id] /= divide_by
