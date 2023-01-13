from logging import Logger
from typing import List, Union

import numpy as np
from tqdm import tqdm

from text_selection_core.globals import TQDM_LINE_UNIT
from text_selection_core.helper import get_float_dtype_from_n, get_int_dtype_from_n
from text_selection_core.types import Dataset, DataWeights, Lines
from text_selection_core.validation import (ErrorType, ValidationErr,
                                            ensure_lines_count_matches_dataset)


def get_from_lines(dataset: Dataset, lines: List[str]) -> Union[ValidationErr, DataWeights]:
  if error := ensure_lines_count_matches_dataset(dataset, lines):
    return error

  try:
    weights = [float(line) for line in lines]
  except ValueError as error:
    return ValidationErr(ErrorType.WEIGHTS_INVALID, error.args[0])

  for weight in weights:
    if not weight >= 0:
      return ValidationErr(ErrorType.WEIGHTS_INVALID, "Weights need to be greater than or equal to zero.")
  all_integer = all(x.is_integer() for x in weights)
  get_max_method = get_int_dtype_from_n if all_integer else get_float_dtype_from_n
  max_n = max(weights)
  result = np.array(weights, dtype=get_max_method(max_n))
  return result


def get_uniform_weights(line_nrs: range, val: int, logger: Logger) -> DataWeights:
  result = np.full(len(line_nrs), fill_value=val, dtype=get_int_dtype_from_n(val))
  #result = dict.fromkeys(line_nrs, 1)
  return result


def get_count_weights(lines: Lines, sep: str, logger: Logger) -> DataWeights:
  if sep == "":
    words_counts = (len(line) for line in lines)
  else:
    words_counts = (
      sum(1 for word in line.split(sep) if word != "")
      for line in lines
    )

  result = list(tqdm(words_counts, desc="Getting counts", unit=TQDM_LINE_UNIT, total=len(lines)))
  max_count = max(result)
  dtype = get_int_dtype_from_n(max_count)
  logger.debug(f"Chosen dtype \"{dtype}\" for numpy because maximum count is {max_count}.")
  result = np.array(result, dtype=dtype)
  logger.debug(f"Min: {np.min(result, axis=0)}")
  logger.debug(f"Mean: {np.mean(result, axis=0)}")
  logger.debug(f"Median: {np.median(result, axis=0)}")
  logger.debug(f"Max: {np.max(result, axis=0)}")
  logger.debug(f"dtype: {result.dtype.name}")
  return result


def divide_weights(weights: DataWeights, divide_by: float, logger: Logger) -> DataWeights:
  assert divide_by > 0
  max_weight = np.max(weights, axis=0)
  dtype = get_float_dtype_from_n(max_weight)
  logger.debug(f"Chosen dtype \"{dtype}\" for numpy because maximum count is {max_weight}.")
  weights = weights.astype(dtype)
  weights = np.divide(weights, divide_by)
  logger.debug(f"Min: {np.min(weights, axis=0)}")
  logger.debug(f"Mean: {np.mean(weights, axis=0)}")
  logger.debug(f"Median: {np.median(weights, axis=0)}")
  logger.debug(f"Max: {np.max(weights, axis=0)}")
  logger.debug(f"dtype: {weights.dtype.name}")
  # for data_id in weights:
  #   weights[data_id] /= divide_by
  return weights
