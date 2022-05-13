
from text_selection_core.types import get_line_nrs


def test_zero():
  result = get_line_nrs(0)
  assert not list(result)
