from typing import Set

from text_selection.common.ngram_extractor import NGram
from text_selection_core.types import (DataIds, Dataset, DataSymbols,
                                       DataWeights, NGramSet, Subset,
                                       SubsetType, Weight)
from text_utils import Symbol


def get_n_grams(symbols: DataSymbols, n_gram: NGram, n_gram_ignore_symbols: Set[Symbol], n_gram_most_common: float, n_gram_least_common: float) -> NGramSet:
  pass
