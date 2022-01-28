import numpy as np
from text_selection.common.ngram_extractor2 import NGramExtractor2


def test_component():
  data = list(
    ("d", "c", "c", "d"),
    ("b", "c", "a", "x"),
    ("a", "b", "a"),
  )

  x = NGramExtractor2(
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
  )

  data_iterator = iter(data)
  x.fit(data_iterator, n_gram=1, ignore_symbols={"c"})

  data_iterator = iter(data)
  result = x.predict(data_iterator, len(data))

  np.testing.assert_array_equal(result, np.array([
    #a, b, d, x
    [0, 0, 2, 0],  # 4
    [2, 1, 0, 0],  # 5
  ]))
