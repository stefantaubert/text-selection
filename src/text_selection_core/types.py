from collections import OrderedDict
from logging import Logger
from typing import Generator, List
from typing import OrderedDict as OrderedDictType
from typing import Union

from ordered_set import OrderedSet

# zero-based line number
LineNr = int
LineNrs = OrderedSet[LineNr]
Line = str
Lines = List[Line]

Percent = float
Weight = Union[float, int]
DataWeights = List[Weight]

Subset = LineNrs
SubsetName = str


def get_line_nrs(line_count) -> range:
  return range(line_count)


class Dataset():
  def __init__(self, line_count: int, default_subset_name: str):
    super().__init__()
    assert line_count > 0
    self.__line_count = line_count
    self.__subsets: OrderedDictType[SubsetName, Subset] = OrderedDict((
      (default_subset_name, OrderedSet(get_line_nrs(line_count))),
    ))

  @property
  def line_count(self) -> int:
    return self.__line_count

  @property
  def subsets(self) -> OrderedDictType[SubsetName, Subset]:
    return self.__subsets

  def get_line_nrs(self) -> range:
    return get_line_nrs(self.__line_count)


def move_lines_to_subset(dataset: Dataset, nrs: LineNrs, target: SubsetName, logger: Logger) -> None:
  logger.debug("Adjusting selection...")
  if target not in dataset.subsets:
    dataset.subsets[target] = OrderedSet()
    logger.debug(f"Created non-existing target subset \"{target}\".")
  target_subset = dataset.subsets[target]
  logger.debug("Update target...")
  target_subset.update(nrs)
  logger.debug("Update subsets...")
  potential_subsets = (dataset.subsets[subset] for subset in dataset.subsets if subset != target)
  for subset in potential_subsets:
    subset.difference_update(nrs)
  logger.debug("Done.")


def get_subsets_line_nrs(dataset: Dataset, subsets: OrderedSet[SubsetName]) -> Generator[LineNr, None, None]:
  from_subsets = (dataset.subsets[from_subset_name] for from_subset_name in subsets)
  lines = (line for subset in from_subsets for line in subset)
  return lines
