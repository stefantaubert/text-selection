from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import cast

from text_selection_core.datasets import create_from_text

from text_selection_app.argparse_helper import (parse_codec,
                                                parse_existing_file,
                                                parse_non_empty_or_whitespace,
                                                parse_path)
from text_selection_app.io import (get_data_symbols_path, get_dataset_path,
                                   save_data_symbols, save_dataset)


def add_encoding_argument(parser: ArgumentParser, help_str: str) -> None:
  parser.add_argument("--encoding", type=parse_codec, metavar='CODEC',
                      help=help_str + "; see all available codecs at https://docs.python.org/3.8/library/codecs.html#standard-encodings", default="utf-8")


def get_dataset_creation_from_text_parser(parser: ArgumentParser):
  parser.description = f"This command reads the lines of a textfile and creates a dataset from it."
  parser.add_argument("directory", type=parse_path, metavar="directory",
                      help="directory to write")
  parser.add_argument("text", type=parse_existing_file, metavar="text",
                      help="path to textfile")
  add_encoding_argument(parser, "encoding of text")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the initial subset containing all Id's", default="base")
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite complete directory")
  return create_dataset_from_text_ns


def create_dataset_from_text_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  data_folder = cast(Path, ns.directory)

  if data_folder.is_dir() and not ns.overwrite:
    logger.error("Directory already exists!")
    return

  lines = cast(Path, ns.text).read_text(ns.encoding).splitlines()

  error, result = create_from_text(lines, ns.name)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
  else:
    dataset, data_symbols = result

    if data_folder.is_dir():
      rmtree(data_folder)

    save_dataset(get_dataset_path(data_folder), dataset)
    save_data_symbols(get_data_symbols_path(data_folder), data_symbols)
