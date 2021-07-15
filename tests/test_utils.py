from text_selection.utils import get_distribution, get_reverse_distribution


def test_get_distribution():
  data = {
    2: [1, 2, 3],
    3: [1, 2],
    4: [1],
  }

  x = get_distribution(data)

  assert_res = {
    1: 3 / 6,
    2: 2 / 6,
    3: 1 / 6,
  }

  assert assert_res == x


def test_get_reverse_distribution():
  data = {
    2: [1, 2, 3],
    3: [1, 2],
    4: [1],
  }

  x = get_reverse_distribution(data)

  assert_res = {
    1: 1 / 6,
    2: 2 / 6,
    3: 3 / 6,
  }

  assert assert_res == x
