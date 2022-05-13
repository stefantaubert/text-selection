# from pathlib import Path
# from shutil import rmtree
# from tempfile import mkdtemp

# from text_selection_cli.io_handling import load_data_symbols, save_data_symbols
# from text_selection_core.types import Lines


# def test_empty__is_empty():
#   tmp_dir = Path(mkdtemp())
#   symbols: Lines = dict()

#   save_data_symbols(tmp_dir, symbols)

#   result = load_data_symbols(tmp_dir)
#   rmtree(tmp_dir)

#   assert result == dict()


# def test_non_empty__creates_file():
#   tmp_dir = Path(mkdtemp())
#   symbols: Lines = dict(((1, ""), (2, "t e s t  a b c.")))

#   save_data_symbols(tmp_dir, symbols)

#   result = load_data_symbols(tmp_dir)
#   rmtree(tmp_dir)

#   assert result == dict(((1, ""), (2, "t e s t  a b c.")))
