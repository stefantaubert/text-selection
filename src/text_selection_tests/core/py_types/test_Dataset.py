from ordered_set import OrderedSet
from text_selection.core.types import Dataset, SubsetType


def test_creates_three_empty_subsets_on_init():
  dataset = Dataset()

  assert len(dataset) == 3
  assert isinstance(dataset.subsets[SubsetType.AVAILABLE], OrderedSet)
  assert isinstance(dataset.subsets[SubsetType.IGNORED], OrderedSet)
  assert isinstance(dataset.subsets[SubsetType.SELECTED], OrderedSet)
  assert len(dataset.subsets[SubsetType.AVAILABLE]) == 0
  assert len(dataset.subsets[SubsetType.IGNORED]) == 0
  assert len(dataset.subsets[SubsetType.SELECTED]) == 0
