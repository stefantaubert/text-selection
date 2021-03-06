# from pathlib import Path
# from shutil import rmtree
# from tempfile import mkdtemp

# from text_selection_cli.io_handling import FILE_EXTENSION, try_save_data_weights
# from text_selection_core.types import DataWeights


# def test_empty__creates_file():
#   tmp_dir = Path(mkdtemp())
#   weights: DataWeights = dict()

#   try_save_data_weights(tmp_dir, weights, "test")

#   assert Path.is_file(tmp_dir / f"test{FILE_EXTENSION}")
#   rmtree(tmp_dir)


# def test_non_empty__creates_file():
#   tmp_dir = Path(mkdtemp())
#   weights: DataWeights = dict(((1, 1.0), (2, 2.0)))

#   try_save_data_weights(tmp_dir, weights, "test")

#   assert Path.is_file(tmp_dir / f"test{FILE_EXTENSION}")
#   rmtree(tmp_dir)


# def test_non_empty__overwrite__overwrites_file():
#   tmp_dir = Path(mkdtemp())
#   weights: DataWeights = dict(((1, 1.0), (2, 2.0)))

#   (tmp_dir / f"test{FILE_EXTENSION}").write_bytes(b"123456")
#   try_save_data_weights(tmp_dir, weights, "test")

#   assert (tmp_dir / f"test{FILE_EXTENSION}").read_bytes() != b"123456"
#   rmtree(tmp_dir)
