from ordered_set import OrderedSet
from text_selection.core.types import Dataset, SubsetType


def test_creates_three_empty_subsets_on_init():
  dataset = Dataset()

  assert len(dataset) == 3
  assert isinstance(dataset[SubsetType.AVAILABLE], OrderedSet)
  assert isinstance(dataset[SubsetType.IGNORED], OrderedSet)
  assert isinstance(dataset[SubsetType.SELECTED], OrderedSet)
  assert len(dataset[SubsetType.AVAILABLE]) == 0
  assert len(dataset[SubsetType.IGNORED]) == 0
  assert len(dataset[SubsetType.SELECTED]) == 0
