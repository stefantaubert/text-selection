# from pathlib import Path
# from shutil import rmtree
# from tempfile import mkdtemp

# from text_selection_cli.io_handling import FILE_EXTENSION, try_save_dataset
# from text_selection_core.types import Dataset


# def test_empty__creates_file():
#   tmp_dir = Path(mkdtemp())
#   dataset = Dataset()

#   try_save_dataset(tmp_dir, dataset, "test")

#   assert Path.is_file(tmp_dir / f"test{FILE_EXTENSION}")
#   rmtree(tmp_dir)


# # def test_non_empty__creates_file():
# #   tmp_dir = Path(mkdtemp())
# #   dataset = Dataset()
# #   dataset.subsets[SubsetType.AVAILABLE] = Subset((1, 3, 2))
# #   dataset.subsets[SubsetType.IGNORED] = Subset((4, 6, 5))
# #   dataset.subsets[SubsetType.SELECTED] = Subset((7, 9, 8))

# #   save_dataset(tmp_dir, dataset, "test")

# #   assert Path.is_file(tmp_dir / f"test{FILE_EXTENSION}")
# #   rmtree(tmp_dir)


# # def test_non_empty__overwrite__overwrites_file():
# #   tmp_dir = Path(mkdtemp())
# #   dataset = Dataset()
# #   dataset.subsets[SubsetType.AVAILABLE] = Subset((1, 3, 2))
# #   dataset.subsets[SubsetType.IGNORED] = Subset((4, 6, 5))
# #   dataset.subsets[SubsetType.SELECTED] = Subset((7, 9, 8))

# #   (tmp_dir / f"test{FILE_EXTENSION}").write_bytes(b"123456")
# #   save_dataset(tmp_dir, dataset, "test")

# #   assert (tmp_dir / f"test{FILE_EXTENSION}").read_bytes() != b"123456"
# #   rmtree(tmp_dir)
