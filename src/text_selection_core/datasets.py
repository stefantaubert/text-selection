from typing import Iterable, Optional, Tuple

from ordered_set import OrderedSet
from text_utils import StringFormat

from text_selection_core.types import Dataset, DataSymbols
from text_selection_core.validation import ValidationError


def create_from_count(count: int, default_subset_name: str) -> Dataset:
  assert count > 0
  result = Dataset(range(count), default_subset_name)
  return result


def create_from_text(lines: Iterable[str], default_subset_name: str) -> Tuple[Optional[ValidationError], Optional[Tuple[Dataset, DataSymbols]]]:
  data_symbols = {
    i: StringFormat.SYMBOLS.convert_symbols_to_string(
      StringFormat.TEXT.convert_string_to_symbols(line))
    for i, line in enumerate(lines)
    if StringFormat.TEXT.can_convert_string_to_symbols(line)
  }

  ids = OrderedSet(data_symbols.keys())

  result = Dataset(ids, default_subset_name)
  return None, (result, data_symbols)
