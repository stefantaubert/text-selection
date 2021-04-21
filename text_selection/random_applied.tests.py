import random
import unittest

from text_selection.random_export import *
from text_selection.utils import *

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def get_random_list(length: int, chars: List[str]) -> List[str]:
  res = [random.choice(chars) for _ in range(length)]
  return res


class UnitTests(unittest.TestCase):

  def test_get_random_seconds_divergence_seeds__is_sorted(self):
    seed = 1111
    random.seed(seed)

    n_data = 100
    data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})

    durations = {k: random.randint(1, 10) for k in data.keys()}

    res, sets = get_random_seconds_divergence_seeds(
      data=data,
      durations_s=durations,
      seconds=60,
      seed=seed,
      samples=1000,
      n=500,
    )

    self.assertEqual(list(res), list(sorted(res)))

  def test_get_random_seconds_divergence_seeds__big_set(self):
    seed = 1111
    random.seed(seed)

    n_data = 10000
    data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})

    durations = OrderedDict({k: random.randint(1, 10) for k in data.keys()})

    res, sets = get_random_seconds_divergence_seeds(
      data=data,
      durations_s=durations,
      seconds=60 * 60,
      seed=seed,
      samples=3000,
      n=2,
    )

    self.assertEqual(OrderedDict({2781, 135}), res)
    self.assertEqual(2, len(sets))


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
