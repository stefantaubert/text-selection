import enum
from typing import Dict, Union

from ordered_set import OrderedSet
from text_utils import SymbolsString

DataId = int
DataIds = OrderedSet[DataId]
DataSymbols = Dict[DataId, SymbolsString]
DataWeights = Dict[DataId, Union[float, int]]

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
