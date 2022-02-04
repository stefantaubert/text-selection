from typing import Iterator
from text_selection_core.types import DataId, DataIds, Subset


def get_fifo_subset_iterator(ids: DataIds) -> Iterator[DataId]:
  return iter(ids)


def get_fifo_original_positions_iterator(subset: Subset, ids: DataIds) -> Iterator[DataId]:
  ordered_subset = ids.intersection(subset)
  return iter(ordered_subset)


def get_fifo_id_iterator(subset: Subset) -> Iterator[DataId]:
  ordered_subset = sorted(subset)
  return iter(ordered_subset)
