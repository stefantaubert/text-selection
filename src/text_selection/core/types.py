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


class Dataset():
  def __init__(self) -> None:
    self.ignored = Subset()
    self.selected = Subset()
    self.available = Subset()


class SubsetType(enum.Enum):
  IGNORED = 0
  SELECTED = 1
  AVAILABLE = 2
