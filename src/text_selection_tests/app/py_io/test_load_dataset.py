from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

from ordered_set import OrderedSet
from text_selection.app.io import load_dataset, save_dataset
from text_selection.core.types import Dataset, Subset


def test_empty__is_dataset_and_ordered_set():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()

  save_dataset(tmp_dir, dataset, "test")

  result = load_dataset(tmp_dir, "test")
  rmtree(tmp_dir)

  assert isinstance(result, Dataset)
  assert isinstance(result.available, OrderedSet)
  assert isinstance(result.ignored, OrderedSet)
  assert isinstance(result.selected, OrderedSet)


def test_empty__contains_nothing():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()

  save_dataset(tmp_dir, dataset, "test")

  result = load_dataset(tmp_dir, "test")
  rmtree(tmp_dir)

  assert len(result.available) == 0
  assert len(result.ignored) == 0
  assert len(result.selected) == 0


def test_non_empty__contains_values():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()
  dataset.available = Subset((1, 3, 2))
  dataset.ignored = Subset((4, 6, 5))
  dataset.selected = Subset((7, 9, 8))

  save_dataset(tmp_dir, dataset, "test")
  result = load_dataset(tmp_dir, "test")
  rmtree(tmp_dir)

  assert result.available == Subset((1, 3, 2))
  assert result.ignored == Subset((4, 6, 5))
  assert result.selected == Subset((7, 9, 8))
