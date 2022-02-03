from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

from ordered_set import OrderedSet
from text_selection_app.io_handling import load_dataset, save_dataset
from text_selection_core.types import Dataset, Subset


def test_empty__is_dataset_and_ordered_set():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()

  save_dataset(tmp_dir, dataset, "test")

  result = load_dataset(tmp_dir, "test")
  rmtree(tmp_dir)

  assert isinstance(result, Dataset)
  assert isinstance(result[SubsetType.AVAILABLE], OrderedSet)
  assert isinstance(result[SubsetType.IGNORED], OrderedSet)
  assert isinstance(result[SubsetType.SELECTED], OrderedSet)


def test_empty__contains_nothing():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()

  save_dataset(tmp_dir, dataset, "test")

  result = load_dataset(tmp_dir, "test")
  rmtree(tmp_dir)

  assert len(result[SubsetType.AVAILABLE]) == 0
  assert len(result[SubsetType.IGNORED]) == 0
  assert len(result[SubsetType.SELECTED]) == 0


def test_non_empty__contains_values():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()
  dataset.subsets[SubsetType.AVAILABLE] = Subset((1, 3, 2))
  dataset.subsets[SubsetType.IGNORED] = Subset((4, 6, 5))
  dataset.subsets[SubsetType.SELECTED] = Subset((7, 9, 8))

  save_dataset(tmp_dir, dataset, "test")
  result = load_dataset(tmp_dir, "test")
  rmtree(tmp_dir)

  assert result[SubsetType.AVAILABLE] == Subset((1, 3, 2))
  assert result[SubsetType.IGNORED] == Subset((4, 6, 5))
  assert result[SubsetType.SELECTED] == Subset((7, 9, 8))
