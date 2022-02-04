from typing import Iterator
from text_selection_core.types import DataId, Subset


def get_reverse_iterator(subset: Subset) -> Iterator[DataId]:
  ordered_subset = reversed(subset)
  return ordered_subset
