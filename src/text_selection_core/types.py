from typing import Dict, Generator, Iterable, Union

import numpy as np
from ordered_set import OrderedSet
from text_selection.common.ngram_extractor import NGram, NGramNr
from text_utils import StringFormat, Symbols, SymbolsString

DataId = int
DataIds = OrderedSet[DataId]
Item = SymbolsString
DataSymbols = Dict[DataId, Item]
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

  def get_subsets_ids(self, subsets: OrderedSet[SubsetName]) -> Generator[DataId, None, None]:
    from_subsets = (self[from_subset_name] for from_subset_name in subsets)
    from_ids = (data_id for subset in from_subsets for data_id in subset)
    return from_ids


class NGramSet():
  def __init__(self) -> None:
    self.data: np.ndarray = None
    self.data_ids_to_indices: Dict[int, DataId] = None
    self.indices_to_data_ids: Dict[DataId, int] = None
    self.n_grams = Dict[NGram, NGramNr]


def item_to_text(item: Item) -> str:
  symbols = StringFormat.SYMBOLS.convert_string_to_symbols(item)
  text = StringFormat.TEXT.convert_symbols_to_string(symbols)
  del symbols
  return text


def item_to_symbols(item: Item) -> Symbols:
  symbols = StringFormat.SYMBOLS.convert_string_to_symbols(item)
  return symbols
