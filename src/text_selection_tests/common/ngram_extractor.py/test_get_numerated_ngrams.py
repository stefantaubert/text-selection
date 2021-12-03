from text_selection.common.ngram_extractor import generate_numerated_ngrams


def test_empty__returs_empty():
  result = generate_numerated_ngrams([])

  assert list(result) == []


def test_two_one_grams():
  result = generate_numerated_ngrams([("a",), ("b",)])

  assert list(result) == [(("a",), 0), (("b",), 1)]
