from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Generator, Iterable
from typing import OrderedDict as OrderedDictType
from typing import Union

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


class Dataset():
  def __init__(self, ids: DataIds):
    self.__ids = ids
    self.__subsets: OrderedDictType[SubsetName, Subset] = OrderedDict()
    super().__init__()

  @property
  def ids(self) -> DataIds:
    return self.__ids

  @property
  def subsets(self) -> OrderedDictType[SubsetName, Subset]:
    return self.__subsets


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


def create_dataset_from_ids(ids: DataIds, default_subset_name: SubsetName) -> Dataset:
  ids = deepcopy(ids)
  res = Dataset(ids)
  res.subsets[default_subset_name] = deepcopy(ids)
  return res


def move_ids_to_subset(dataset: Dataset, ids: DataIds, target: SubsetName) -> None:
  assert target in dataset.subsets
  target_subset = dataset.subsets[target]
  target_subset |= ids

  potential_subsets = (dataset.subsets[subset] for subset in dataset.subsets if subset != target)
  for subset in potential_subsets:
    subset -= ids


def get_subsets_ids(dataset: Dataset, subsets: OrderedSet[SubsetName]) -> Generator[DataId, None, None]:
  from_subsets = (dataset.subsets[from_subset_name] for from_subset_name in subsets)
  from_ids = (data_id for subset in from_subsets for data_id in subset)
  return from_ids
