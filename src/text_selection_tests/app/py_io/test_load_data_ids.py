# from pathlib import Path
# from shutil import rmtree
# from tempfile import mkdtemp

# from ordered_set import OrderedSet
# from text_selection_app.io import load_data_ids, save_data_ids
# from text_selection_core.types import DataIds


# def test_empty__is_empty():
#   tmp_dir = Path(mkdtemp())
#   ids = DataIds()
#   save_data_ids(tmp_dir, ids)

#   result = load_data_ids(tmp_dir)
#   rmtree(tmp_dir)

#   assert result == DataIds()


# def test_non_empty__is_ordered_set():
#   tmp_dir = Path(mkdtemp())
#   ids = DataIds()
#   save_data_ids(tmp_dir, ids)

#   result = load_data_ids(tmp_dir)
#   rmtree(tmp_dir)

#   assert isinstance(result, OrderedSet)


# def test_non_empty__keeps_order():
#   tmp_dir = Path(mkdtemp())
#   ids = DataIds((1, 4, 2, 0, 7))
#   save_data_ids(tmp_dir, ids)

#   result = load_data_ids(tmp_dir)
#   rmtree(tmp_dir)

#   assert result == DataIds((1, 4, 2, 0, 7))

