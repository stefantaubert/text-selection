
from argparse import ArgumentParser
from multiprocessing import cpu_count

from text_selection_cli.argparse_helper import (ConvertToOrderedSetAction, get_optional,
                                                parse_codec, parse_existing_file, parse_non_empty,
                                                parse_non_empty_or_whitespace,
                                                parse_positive_integer)

DEFAULT_N_JOBS = cpu_count()
DEFAULT_CHUNKSIZE = 500
DEFAULT_MAXTASKSPERCHILD = None

parse_weights_name = parse_non_empty_or_whitespace


def add_from_and_to_subsets_arguments(parser: ArgumentParser) -> None:
  add_from_subsets_argument(parser)
  add_to_subset_argument(parser)


def add_from_subsets_argument(parser: ArgumentParser) -> None:
  parser.add_argument("from_subsets", type=parse_non_empty_or_whitespace, nargs="+",
                      metavar="from-subsets", help="from subset", action=ConvertToOrderedSetAction)


def add_to_subset_argument(parser: ArgumentParser) -> None:
  parser.add_argument("to_subset", type=parse_non_empty_or_whitespace,
                      metavar="to-subset", help="to subset")


def add_project_argument(parser: ArgumentParser) -> None:
  parser.add_argument("project", type=parse_existing_file, metavar="project",
                      help="directory containing project data")


def add_file_arguments(parser: ArgumentParser, include_sep: bool = False) -> None:
  parser.add_argument("file", type=parse_non_empty_or_whitespace,
                      help="name of the file containing the lines")
  parser.add_argument("--lsep", type=parse_non_empty, default="\n",
                      help="line separator")
  if include_sep:
    add_sep_argument(parser)
  add_encoding_argument(parser, "encoding of file")


def add_sep_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--sep", type=str, metavar="SYMBOL",
                      help="separator symbol for symbols/words", default=" ")


def add_n_jobs_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-j", "--n-jobs", metavar='N', type=int,
                      choices=range(1, cpu_count() + 1), default=DEFAULT_N_JOBS, help="amount of parallel cpu jobs")


def add_chunksize_argument(parser: ArgumentParser, target: str = "files", default: int = DEFAULT_CHUNKSIZE) -> None:
  parser.add_argument("-s", "--chunksize", type=parse_positive_integer, metavar="NUMBER",
                      help=f"amount of {target} to chunk into one job", default=default)


def add_maxtaskperchild_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-m", "--maxtasksperchild", type=get_optional(parse_positive_integer), metavar="NUMBER",
                      help="amount of tasks per child", default=DEFAULT_MAXTASKSPERCHILD)


# def add_string_format_argument(parser: ArgumentParser, target: str, short_name: str = "-f", name: str = '--formatting') -> None:
#   names = OrderedDict((
#     (StringFormat2.DEFAULT, "Normal"),
#     (StringFormat2.SPACED, "Spaced"),
#   ))

#   values_to_names = dict(zip(
#     names.values(),
#     names.keys()
#   ))

#   help_str = f"formatting of text in {target}; use \'{names[StringFormat2.DEFAULT]}\' for normal text and \'{names[StringFormat2.SPACED]}\' for space separated symbols, i.e., words are separated by two spaces and characters are separated by one space. Example: {names[StringFormat2.DEFAULT]} -> |This text.|; {names[StringFormat2.SPACED]} -> |T␣h␣i␣s␣␣t␣e␣x␣t␣.|"
#   parser.add_argument(
#     short_name, name,
#     metavar=list(names.values()),
#     choices=StringFormat2,
#     type=values_to_names.get,
#     default=names[StringFormat2.DEFAULT],
#     help=help_str,
#   )


def add_encoding_argument(parser: ArgumentParser, help_str: str) -> None:
  parser.add_argument("--encoding", type=parse_codec, metavar='CODEC',
                      help=help_str + "; see all available codecs at https://docs.python.org/3.8/library/codecs.html#standard-encodings", default="UTF-8")
