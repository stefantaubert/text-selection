
from text_utils import StringFormat
from argparse import ArgumentParser
from collections import OrderedDict
from multiprocessing import cpu_count

from text_selection_app.argparse_helper import (get_optional, parse_codec,
                                                parse_positive_integer)

DEFAULT_N_JOBS = cpu_count()
DEFAULT_CHUNKSIZE = 500
DEFAULT_MAXTASKSPERCHILD = None


def add_n_jobs_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-j", "--n-jobs", metavar='N', type=int,
                      choices=range(1, cpu_count() + 1), default=DEFAULT_N_JOBS, help="amount of parallel cpu jobs")


def add_chunksize_argument(parser: ArgumentParser, target: str = "files", default: int = DEFAULT_CHUNKSIZE) -> None:
  parser.add_argument("-s", "--chunksize", type=parse_positive_integer, metavar="NUMBER",
                      help=f"amount of {target} to chunk into one job", default=default)


def add_maxtaskperchild_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-m", "--maxtasksperchild", type=get_optional(parse_positive_integer), metavar="NUMBER",
                      help="amount of tasks per child", default=DEFAULT_MAXTASKSPERCHILD)


def add_string_format_argument(parser: ArgumentParser, target: str, short_name: str = "-f", name: str = '--formatting') -> None:
  names = OrderedDict((
    (StringFormat.TEXT, "Normal"),
    (StringFormat.SYMBOLS, "Spaced"),
  ))

  values_to_names = dict(zip(
    names.values(),
    names.keys()
  ))

  help_str = f"formatting of text in {target}; use \'{names[StringFormat.TEXT]}\' for normal text and \'{names[StringFormat.SYMBOLS]}\' for space separated symbols, i.e., words are separated by two spaces and characters are separated by one space. Example: {names[StringFormat.TEXT]} -> |This text.|; {names[StringFormat.SYMBOLS]} -> |T␣h␣i␣s␣␣t␣e␣x␣t␣.|"
  parser.add_argument(
    short_name, name,
    metavar=list(names.values()),
    choices=StringFormat,
    type=values_to_names.get,
    default=names[StringFormat.TEXT],
    help=help_str,
  )


def add_encoding_argument(parser: ArgumentParser, help_str: str) -> None:
  parser.add_argument("--encoding", type=parse_codec, metavar='CODEC',
                      help=help_str + "; see all available codecs at https://docs.python.org/3.8/library/codecs.html#standard-encodings", default="UTF-8")
