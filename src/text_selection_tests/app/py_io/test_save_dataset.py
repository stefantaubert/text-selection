from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

from text_selection.app.io import FILE_EXTENSION, save_dataset
from text_selection.core.types import Dataset, Subset


def test_empty__creates_file():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()

  save_dataset(tmp_dir, dataset, "test")

  assert Path.is_file(tmp_dir / f"test{FILE_EXTENSION}")
  rmtree(tmp_dir)


def test_non_empty__creates_file():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()
  dataset.available = Subset((1, 3, 2))
  dataset.ignored = Subset((4, 6, 5))
  dataset.selected = Subset((7, 9, 8))

  save_dataset(tmp_dir, dataset, "test")

  assert Path.is_file(tmp_dir / f"test{FILE_EXTENSION}")
  rmtree(tmp_dir)


def test_non_empty__overwrite__overwrites_file():
  tmp_dir = Path(mkdtemp())
  dataset = Dataset()
  dataset.available = Subset((1, 3, 2))
  dataset.ignored = Subset((4, 6, 5))
  dataset.selected = Subset((7, 9, 8))

  (tmp_dir / f"test{FILE_EXTENSION}").write_bytes(b"123456")
  save_dataset(tmp_dir, dataset, "test")

  assert (tmp_dir / f"test{FILE_EXTENSION}").read_bytes() != b"123456"
  rmtree(tmp_dir)
