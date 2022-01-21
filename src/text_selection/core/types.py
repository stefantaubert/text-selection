import enum
from typing import Dict, Union

import numpy as np
from ordered_set import OrderedSet
from text_selection.common.ngram_extractor import NGram
from text_utils import SymbolsString

DataId = int
DataIds = OrderedSet[DataId]
DataSymbols = Dict[DataId, SymbolsString]
Weight = Union[float, int]
DataWeights = Dict[DataId, Weight]

Selection = OrderedSet[DataId]
Subset = Selection


class SubsetType(enum.Enum):
  IGNORED = 0
  SELECTED = 1
  AVAILABLE = 2


class Dataset(Dict[SubsetType, Subset]):
  def __init__(self) -> None:
    self[SubsetType.AVAILABLE] = Subset()
    self[SubsetType.SELECTED] = Subset()
    self[SubsetType.IGNORED] = Subset()


class NGramSet():
  def __init__(self) -> None:
    self.data: np.ndarray = None
    self.data_ids: Dict[int, DataId] = None
    self.n_grams = Dict[int, NGram]
