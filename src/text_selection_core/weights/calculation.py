from logging import Logger

import numpy as np
from tqdm import tqdm

from text_selection_core.helper import get_dtype_from_count
from text_selection_core.types import DataWeights, Lines


def get_uniform_weights(line_nrs: range) -> DataWeights:
  result = np.full(len(line_nrs), fill_value=1, dtype=np.uint8)
  #result = dict.fromkeys(line_nrs, 1)
  return result


def get_word_count_weights(lines: Lines, sep: str, logger: Logger) -> DataWeights:
  if sep == "":
    words_counts = (len(line) for line in lines)
  else:
    words_counts = (
      sum(1 for word in line.split(sep) if word != "")
      for line in lines
    )

  result = list(tqdm(words_counts, desc="Getting counts", unit=" line(s)", total=len(lines)))
  max_count = max(result)
  dtype = get_dtype_from_count(max_count)
  logger.debug(f"Chosen dtype \"{dtype}\" for numpy because maximum count is {max_count}.")
  result = np.array(result, dtype=dtype)
  return result


def divide_weights_inplace(weights: DataWeights, divide_by: float) -> None:
  assert divide_by > 0
  weights /= divide_by
  # for data_id in weights:
  #   weights[data_id] /= divide_by
