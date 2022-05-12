from typing import Iterable, Optional, Tuple

from ordered_set import OrderedSet
from text_utils import StringFormat2

from text_selection_core.types import (Dataset, DataSymbols,
                                       create_dataset_from_ids)
from text_selection_core.validation import ValidationError


def create_from_count(count: int, default_subset_name: str) -> Dataset:
  assert count > 0
  result = create_dataset_from_ids(range(count), default_subset_name)
  return result

from tqdm import tqdm


def create_from_text(lines: Iterable[str], default_subset_name: str, string_format: StringFormat2) -> Tuple[Optional[ValidationError], Optional[Tuple[Dataset, DataSymbols]]]:
  data_symbols = {
    i: StringFormat2.SPACED.convert_symbols_to_string(
      string_format.convert_string_to_symbols(line))
    for i, line in enumerate(tqdm(lines))
    if string_format.can_convert_string_to_symbols(line)
  }

  ids = OrderedSet(data_symbols.keys())

  result = create_dataset_from_ids(ids, default_subset_name)
  return None, (result, data_symbols)
