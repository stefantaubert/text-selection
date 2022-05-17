
from typing import List

import numpy as np

from text_selection_core.types import DataWeights, Percent, Subset, Weight

dtype_order_uint = [
  np.uint8,
  np.uint16,
  np.uint32,
  np.uint64,
  # np.uint128,
  # np.uint256,
]


def get_dtype_from_count(count: int) -> np.dtype:
  for dtype in dtype_order_uint:
    if count < np.iinfo(dtype).max:
      return dtype
  raise ValueError("Parameter 'count' to big for numpy!")


def get_target_weights_from_percent(from_subset: Subset, to_subset: Subset, weights: DataWeights, target: Percent, target_incl_selection: bool) -> Weight:
  if target_incl_selection:
    target_value = target * sum(weights[k] for k in from_subset | to_subset)
  else:
    target_value = target * sum(weights[k] for k in from_subset)
  return target_value


def get_initial_weights(to_subset: Subset, weights: DataWeights, target_incl_selection: bool) -> Weight:
  if target_incl_selection:
    initial_weights = sum(weights[k] for k in to_subset)
  else:
    initial_weights = 0
  return initial_weights


def move_selection_between_subsets(from_subset: Subset, from_selection: Subset, to_subset: Subset) -> None:
  assert from_selection.issubset(from_subset)
  assert len(from_subset.intersection(to_subset)) == 0

  to_subset |= from_selection
  from_subset -= from_selection


def split_adv(s: str, sep: str) -> List[str]:
  if sep == "":
    return list(s)
  return s.split(sep)
