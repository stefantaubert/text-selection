
from typing import Optional, Set

from text_selection.common.ngram_extractor import NGram
from text_selection_core.types import (DataIds, Dataset, DataSymbols,
                                       DataWeights, NGramSet, Subset,
                                       SubsetType, Weight)
from text_utils import Symbol


def select_greedy_weight_based_absolute(from_subset: Subset, to_subset: Subset, n_grams: NGramSet, weights: DataWeights, consider_to_subset: bool, target: Weight, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int):
  # TODO assert dataset keys == ids in app; assert from_subset has no intersection with to_subset
  pass
