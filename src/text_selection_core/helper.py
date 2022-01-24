from math import ceil
from typing import Optional, Set

from ordered_set import OrderedSet
from text_selection.common.ngram_extractor import NGram
from text_utils import Symbol

from text_selection_core.types import (DataIds, Dataset, DataSymbols,
                                       DataWeights, NGramSet, Percent, Subset,
                                       Weight)
from text_selection_core.weights.weights_iterator import WeightsIterator


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
