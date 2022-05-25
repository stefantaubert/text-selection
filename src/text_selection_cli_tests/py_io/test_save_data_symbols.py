# from pathlib import Path
# from shutil import rmtree
# from tempfile import mkdtemp

# from text_selection_cli.io_handling import DATA_SYMBOLS_NAME, FILE_EXTENSION, save_data_symbols
# from text_selection_core.types import Lines


# def test_empty__creates_file():
#   tmp_dir = Path(mkdtemp())
#   symbols: Lines = dict()

#   save_data_symbols(tmp_dir, symbols)

#   assert Path.is_file(tmp_dir / f"{DATA_SYMBOLS_NAME}{FILE_EXTENSION}")
#   rmtree(tmp_dir)


# def test_non_empty__creates_file():
#   tmp_dir = Path(mkdtemp())
#   symbols: Lines = dict(((1, ""), (2, "t e s t  a b c.")))

#   save_data_symbols(tmp_dir, symbols)

#   assert Path.is_file(tmp_dir / f"{DATA_SYMBOLS_NAME}{FILE_EXTENSION}")
#   rmtree(tmp_dir)


# def test_non_empty__overwrite__overwrites_file():
#   tmp_dir = Path(mkdtemp())
#   symbols: Lines = dict(((1, ""), (2, "t e s t  a b c.")))

#   (tmp_dir / f"{DATA_SYMBOLS_NAME}{FILE_EXTENSION}").write_bytes(b"123456")
#   save_data_symbols(tmp_dir, symbols)

#   assert (tmp_dir / f"{DATA_SYMBOLS_NAME}{FILE_EXTENSION}").read_bytes() != b"123456"
#   rmtree(tmp_dir)
