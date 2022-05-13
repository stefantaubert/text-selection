# from logging import getLogger
# from pathlib import Path
# from shutil import rmtree
# from tempfile import mkdtemp

# from text_selection_cli.io_handling import try_load_data_weights, try_save_data_weights
# from text_selection_core.types import DataWeights


# def test_empty__is_empty():
#   tmp_dir = Path(mkdtemp())
#   weights: DataWeights = dict()

#   try_save_data_weights(tmp_dir, weights, "test", getLogger())

#   result = try_load_data_weights(tmp_dir, "test")
#   rmtree(tmp_dir)

#   assert result == dict()


# def test_non_empty__creates_file():
#   tmp_dir = Path(mkdtemp())
#   weights: DataWeights = dict(((1, 1.0), (2, 2.0)))

#   try_save_data_weights(tmp_dir, weights, "test", getLogger())

#   result = try_load_data_weights(tmp_dir, "test")
#   rmtree(tmp_dir)

#   assert result == dict(((1, 1.0), (2, 2.0)))
