import unittest

from text_selection.cover_export import *
from text_selection.utils import *


class UnitTests(unittest.TestCase):
  def test_cover_symbols_default(self):
    data = OrderedDict({
      1: ["a", "a"],  # one new
      2: ["c", "a"],  # two new
      3: ["d", "a"],  # two new
    })

    res = cover_symbols_default(
      data=data,
      symbols={"c"},
    )

    self.assertEqual(OrderedSet([2]), res)

  def test_cover_symbols_default__not_existing_symbols__are_ignored(self):
    data = OrderedDict({
      1: ["a", "a"],  # one new
      2: ["c", "a"],  # two new
      3: ["d", "a"],  # two new
    })

    res = cover_symbols_default(
      data=data,
      symbols={"x"},
    )

    self.assertEqual(OrderedSet([]), res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
