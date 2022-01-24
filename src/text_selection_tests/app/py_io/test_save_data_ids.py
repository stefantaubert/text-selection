from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

from text_selection.app.io import DATA_IDS_NAME, FILE_EXTENSION, save_data_ids
from text_selection.core.types import DataIds


def test_empty__creates_file():
  tmp_dir = Path(mkdtemp())
  ids = DataIds()
  save_data_ids(tmp_dir, ids)
  assert Path.is_file(tmp_dir / f"{DATA_IDS_NAME}{FILE_EXTENSION}")
  rmtree(tmp_dir)


def test_non_empty__creates_file():
  tmp_dir = Path(mkdtemp())
  ids = DataIds((1, 2, 4))
  save_data_ids(tmp_dir, ids)
  assert Path.is_file(tmp_dir / f"{DATA_IDS_NAME}{FILE_EXTENSION}")
  rmtree(tmp_dir)


def test_non_empty__overwrite__overwrites_file():
  tmp_dir = Path(mkdtemp())
  ids = DataIds((1, 2, 4))

  (tmp_dir / f"{DATA_IDS_NAME}{FILE_EXTENSION}").write_bytes(b"123456")
  save_data_ids(tmp_dir, ids)

  assert (tmp_dir / f"{DATA_IDS_NAME}{FILE_EXTENSION}").read_bytes() != b"123456"
  rmtree(tmp_dir)
