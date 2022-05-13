from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

from ordered_set import OrderedSet
from tqdm import tqdm

from txt_selection_cli.globals import ExecutionResult, SEL_EXT
from txt_selection_cli.helper import (ConvertToOrderedSetAction, add_encoding_argument,
                                      get_all_files_in_all_subfolders, parse_existing_directory,
                                      parse_non_empty, parse_non_empty_or_whitespace, try_save_selection)
from txt_selection_cli.logging_configuration import get_file_logger, init_and_get_console_logger


def get_initialization_parser(parser: ArgumentParser):
  parser.description = "This command merges multiple files into one."
  parser.add_argument("directories", type=parse_existing_directory,
                      metavar="directory", nargs="+", help="directories containing text files", action=ConvertToOrderedSetAction)
  parser.add_argument("file", type=parse_non_empty_or_whitespace,
                      help="file name containing the data lines, e.g., lines.txt")
  parser.add_argument("name", type=parse_non_empty_or_whitespace,
                      help="name of the initial set, e.g., base")
  parser.add_argument("--lsep", type=parse_non_empty, default="\n",
                      help="line separator")
  add_encoding_argument(parser, "encoding of the file")
  return init_ns


def init_ns(ns: Namespace) -> ExecutionResult:
  logger = init_and_get_console_logger(__name__)
  flogger = get_file_logger()

  if ns.name == ns.file:
    logger.error("Parameter 'file' and 'name' need to be distinct!")
    return False, None

  logger.info("Searching for files...")
  files: List[Path] = list(
    file
    for folder in ns.directories
    for file in get_all_files_in_all_subfolders(folder)
    if file.name == ns.file
  )

  if len(files) == 0:
    logger.info("Did not find any matching files!")
    return True, None

  all_successfull = True
  path: Path
  for path in tqdm(files, desc="Initializing folder(s)", unit=" folder(s)"):
    flogger.info(f"Processing \"{path}\"")
    selection_path = path.parent / f"{ns.name}{SEL_EXT}"
    if selection_path.exists():
      flogger.error(f"Folder \"{path.parent}\" is already initialized! Skipped")
      all_successfull = False
      continue
    flogger.info("Reading text...")
    try:
      text = path.read_text(ns.encoding)
    except Exception as ex:
      flogger.error(f"File \"{path.absolute()}\" couldn't be loaded!")
      flogger.exception(ex)
      all_successfull = False
      continue
    flogger.info("Splitting lines...")
    lines = text.split(ns.lsep)
    lines_count = len(lines)
    del text
    del lines
    flogger.info(f"Parsed {lines_count} lines.")

    selection = OrderedSet(range(1, lines_count + 1))
    success = try_save_selection(selection, selection_path, flogger)
    if not success:
      all_successfull = False
      continue

  return all_successfull, None
