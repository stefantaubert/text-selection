import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Generator, List, cast

from ordered_set import OrderedSet
from tqdm import tqdm

from txt_selection_cli.globals import SEL_ENC, SEL_EXT, SEL_LSEP, ExecutionResult
from txt_selection_cli.helper import (ConvertToOrderedSetAction, add_encoding_argument,
                                      get_all_files_in_all_subfolders, parse_existing_directory,
                                      parse_non_empty, parse_non_empty_or_whitespace, parse_path)
from txt_selection_cli.logging_configuration import get_file_logger, init_and_get_console_logger


def get_deinitialization_parser(parser: ArgumentParser):
  parser.description = "This command merges multiple files into one."
  parser.add_argument("directories", type=parse_existing_directory,
                      metavar="directory", nargs="+", help="directories containing selection files", action=ConvertToOrderedSetAction)
  return deinit_ns


def deinit_ns(ns: Namespace) -> ExecutionResult:
  logger = init_and_get_console_logger(__name__)
  flogger = get_file_logger()

  logger.info("Searching for selection files...")
  sel_files: List[Path] = list(
    file
    for folder in ns.directories
    for file in get_all_files_in_all_subfolders(folder)
    if file.suffix.lower() == SEL_EXT.lower()
  )

  if len(sel_files) == 0:
    logger.info("Did not find any selection files!")
    return True, None

  all_successfull = True
  path: Path
  for path in tqdm(sel_files, desc="Removing selection file(s)", unit=" file(s)"):
    flogger.info(f"Removing \"{path}\"")
    try:
      os.remove(path)
    except Exception as ex:
      flogger.error("File couldn't be removed!")
      flogger.exception(ex)
      all_successfull = False
      continue
  return all_successfull, None
