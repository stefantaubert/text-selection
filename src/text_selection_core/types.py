from collections import OrderedDict
from copy import deepcopy
from logging import Logger, getLogger
from typing import Dict, Generator, Iterable, List
from typing import OrderedDict as OrderedDictType
from typing import Union

import numpy as np
from ordered_set import OrderedSet

LineNr = int
LineNrs = OrderedSet[LineNr]
Line = str
Lines = List[Line]

Percent = float
Weight = Union[float, int]
DataWeights = List[Weight]

Subset = LineNrs
SubsetName = str


class Dataset():
  def __init__(self, nrs: LineNrs):
    self.__nrs = nrs
    self.__subsets: OrderedDictType[SubsetName, Subset] = OrderedDict()
    super().__init__()

  @property
  def nrs(self) -> LineNrs:
    return self.__nrs

  @property
  def subsets(self) -> OrderedDictType[SubsetName, Subset]:
    return self.__subsets


def create_dataset_from_line_count(count: int, default_subset_name: SubsetName) -> Dataset:
  subset = OrderedSet(range(1, count + 1))
  res = Dataset(subset)
  res.subsets[default_subset_name] = deepcopy(subset)
  return res


def move_lines_to_subset(dataset: Dataset, nrs: LineNrs, target: SubsetName, logger: Logger) -> None:
  logger.debug("Adjusting selection...")
  assert target in dataset.subsets
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
