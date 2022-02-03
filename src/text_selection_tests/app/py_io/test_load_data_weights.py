from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

from text_selection_app.io_handling import load_data_weights, save_data_weights
from text_selection_core.types import DataWeights


def test_empty__is_empty():
  tmp_dir = Path(mkdtemp())
  weights: DataWeights = dict()

  save_data_weights(tmp_dir, weights, "test")

  result = load_data_weights(tmp_dir, "test")
  rmtree(tmp_dir)

  assert result == dict()


def test_non_empty__creates_file():
  tmp_dir = Path(mkdtemp())
  weights: DataWeights = dict(((1, 1.0), (2, 2.0)))

  save_data_weights(tmp_dir, weights, "test")

  result = load_data_weights(tmp_dir, "test")
  rmtree(tmp_dir)

  assert result == dict(((1, 1.0), (2, 2.0)))
