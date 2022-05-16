import numpy as np
from tqdm import tqdm

from text_selection_core.types import DataWeights, Lines


def get_uniform_weights(line_nrs: range) -> DataWeights:
  result = np.full(len(line_nrs), fill_value=1)
  #result = dict.fromkeys(line_nrs, 1)
  return result


def get_word_count_weights(lines: Lines, sep: str) -> DataWeights:
  texts = (
    (line_nr, line)
    for line_nr, line in enumerate(lines)
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

  result = list(tqdm(words_counts, desc="Getting counts", unit=" line(s)", total=len(lines)))
  result = np.array(result)
  return result


def divide_weights_inplace(weights: DataWeights, divide_by: float) -> None:
  assert divide_by > 0
  weights /= divide_by
  # for data_id in weights:
  #   weights[data_id] /= divide_by
