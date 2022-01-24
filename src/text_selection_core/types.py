from typing import Dict, Iterable, Union

import numpy as np
from ordered_set import OrderedSet
from text_selection.common.ngram_extractor import NGram
from text_utils import SymbolsString

DataId = int
DataIds = OrderedSet[DataId]
DataSymbols = Dict[DataId, SymbolsString]
Percent = float
Weight = Union[float, int]
DataWeights = Dict[DataId, Weight]

Subset = DataIds
SubsetName = str


class Dataset(Dict[SubsetName, Subset]):
  def __init__(self, ids: DataIds, default_subset_name: SubsetName) -> None:
    self.ids = ids.copy()
    self[default_subset_name] = ids.copy()

  def get_subset_from_id(self, data_id: DataId) -> Subset:
    assert data_id in self.ids
    for subset in self.values():
      if data_id in subset:
        return subset
    assert False

  def move_ids_to_subset(self, ids: Iterable[DataId], target: SubsetName) -> None:
    assert target in self
    target_subset = self[target]
    for data_id in ids:
      source_subset = self.get_subset_from_id(data_id)
      source_subset.remove(data_id)
      target_subset.add(data_id)


class NGramSet():
  def __init__(self) -> None:
    self.data: np.ndarray = None
    self.data_ids: Dict[int, DataId] = None
    self.n_grams = Dict[int, NGram]
